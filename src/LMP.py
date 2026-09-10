import os
import openai
openai.base_url = "https://api.deepseek.com"
from time import sleep
try:
    from openai.error import RateLimitError, APIConnectionError
except ImportError:
    try:
        from openai import RateLimitError, APIConnectionError
    except ImportError:
        RateLimitError = Exception
        APIConnectionError = Exception
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation, bcolors
import time
from LLM_cache import DiskCache

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "璇疯緭鍏ユ偍鑷繁鐨凞eepSeek-API-Key(sk-...)  Set OPENAI_API_KEY env var"),
    base_url="https://api.deepseek.com"
)

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        is_chat_model = kwargs['model'] != 'gpt-3.5-turbo-instruct' and \
            any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4', 'deepseek']])
        if is_chat_model:
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query
            # handle given context (this was written originally for completion endpoint)
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                # remove obj_context from user1
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                # add obj_context to user2
                user2 = obj_context.strip() + '\n' + user2
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            kwargs['messages'] = messages
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                response = client.chat.completions.create(
                    model=kwargs['model'],
                    messages=messages,
                    temperature=kwargs.get('temperature', 0),
                    max_tokens=kwargs.get('max_tokens', 512),
                )
                ret = response.choices[0].message.content
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs]
            else:
                ret = openai.Completion.create(**kwargs)['choices'][0]['text'].strip()
                self._cache[kwargs] = ret
                return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            _exec_ok = True
            try:
                exec_safe(to_exec, gvars, lvars)
            except Exception as _exec_err:
                _exec_ok = False
                print(f'{bcolors.WARNING}[LMP.py | {self._name}] Main generated code raised during execution: {type(_exec_err).__name__}: {_exec_err}{bcolors.ENDC}')
                try:
                    import traceback as _tb_exec
                    print(_tb_exec.format_exc()[-900:])
                except Exception:
                    pass
                # For planner/composer, do NOT give up: fall through to generic action heuristics
                # (LLM often returns incomplete / name-error snippets for rare color adjectives)
                if self._name in ['planner', 'composer']:
                    print(f'{bcolors.WARNING}[LMP.py | {self._name}] Will try generic fallback action to recover from exec error{bcolors.ENDC}')
                else:
                    # Non-action LMPs: re-raise (fail fast; caller should decide)
                    raise
            # === Composer fallback: if LLM returned empty code / comment-only, or raised NameError
            #     (incomplete code snippet from LLM), run hardcoded action heuristics ===
            # --- Skip fallback for reset / "back to default" tail queries (they legitimately have no execute()) ---
            _q_lwr = (query or '').strip().lower()
            _is_reset_tail = (
                ('reset_to_default_pose' in to_exec) or
                ('back to default' in _q_lwr) or
                ('default pose' in _q_lwr) or
                ('reset' == _q_lwr) or
                _q_lwr.startswith('reset ') or
                ('to default' in _q_lwr)
            )
            _no_action = (
                (self._name in ['composer', 'planner']) and
                (not _is_reset_tail) and
                (
                    # Case A: no action-call present AND exec finished cleanly (LLM wrote comment-only code)
                    (('execute(' not in to_exec) and ('composer(' not in to_exec) and _exec_ok) or
                    # Case B: any exec error (regardless of whether execute() was syntactically
                    # present) — the execute() call inside generated code might raise AFTER
                    # running for half the pipeline and we cannot rely on partial state.
                    (not _exec_ok)
                )
            )
            # ── Wrong-colored object detection ─────────────────────────────────────────
            # Scenario: composer exec_ok = True, parse_query_obj('rose button') returned
            # generic detect('button') WITHOUT color-checking (LLM just wrote
            # "ret_val = button"). The returned object is the WRONG button (e.g., violet
            # instead of rose).  The fallback code knows how to scan ALL color-synonym
            # variants of "<color> button" entries via env detect and pick the correct one.
            # So we must treat silent wrong-color as a recovery trigger.
            _wrong_color = False
            if self._name == 'composer' and _exec_ok and (not _is_reset_tail):
                try:
                    _q_lwr2 = _q_lwr
                    _COLOR_SYN_SET = set([
                        'purple','violet','magenta','pink','red','crimson','maroon','coral','salmon',
                        'orange','brown','tan','beige','cream','yellow','gold','olive','lime','green',
                        'teal','aqua','cyan','turquoise','azure','blue','navy','indigo','black','white',
                        'gray','grey','silver','rose',
                    ])
                    # Extract color adjectives + noun from query (e.g., "push the rose button" → rose)
                    _q_toks = _re_fb_xx.findall(r"[a-z0-9_]+", _q_lwr2) if '_re_fb_xx' in dir() else _q_lwr2.split()
                    import re as _re_fb_xx_h
                    _q_toks = _re_fb_xx_h.findall(r"[a-z0-9_]+", _q_lwr2)
                    _tok_colors = [t for t in _q_toks if t in _COLOR_SYN_SET]
                    if len(_tok_colors) > 0:
                        # Try to read the actual movable that was set during exec (if any)
                        _suspect_movable = None
                        try:
                            _suspect_movable = lvars.get('movable', None)
                            if _suspect_movable is None and 'ret_val' in lvars:
                                _suspect_movable = lvars.get('ret_val', None)
                        except Exception:
                            _suspect_movable = None
                        if _suspect_movable is not None:
                            # Read .color attribute safely (it may be wrapped DynamicObservation)
                            _obj_color = None
                            try:
                                _obj_color = _suspect_movable.color
                                if callable(_obj_color):
                                    try: _obj_color = _obj_color()
                                    except Exception: _obj_color = None
                            except Exception:
                                pass
                            try:
                                if _obj_color is None and isinstance(_suspect_movable, dict):
                                    _obj_color = _suspect_movable.get('color', None)
                            except Exception:
                                pass
                            _obj_color_clean = str(_obj_color or '').strip().lower()
                            # Build synonym-expanded set for user's instruction color adj
                            _syn_table = {
                                'purple': {'purple','violet','magenta','pink','indigo','crimson'},
                                'violet': {'violet','purple','indigo','magenta'},
                                'magenta': {'magenta','purple','pink','violet','crimson','red'},
                                'pink': {'pink','magenta','purple','coral','salmon','red','rose'},
                                'rose': {'rose','pink','magenta','red','coral','salmon'},
                                'red': {'red','crimson','maroon','magenta','pink','coral','salmon','rose'},
                                'crimson': {'crimson','red','maroon','magenta','purple'},
                                'maroon': {'maroon','red','crimson','brown'},
                                'coral': {'coral','orange','salmon','pink','red'},
                                'salmon': {'salmon','pink','coral','orange','red'},
                                'orange': {'orange','coral','salmon','gold','yellow','brown'},
                                'brown': {'brown','orange','maroon','tan','beige','gold'},
                                'tan': {'tan','brown','beige','gold','orange'},
                                'beige': {'beige','tan','cream','white','gold','brown'},
                                'cream': {'cream','beige','white','yellow','tan'},
                                'yellow': {'yellow','gold','orange','cream','beige','lime'},
                                'gold': {'gold','yellow','orange','brown','tan','beige'},
                                'olive': {'olive','green','yellow','lime','brown'},
                                'lime': {'lime','green','yellow','olive','aqua'},
                                'green': {'green','lime','olive','teal','aqua','cyan','turquoise'},
                                'teal': {'teal','green','cyan','turquoise','aqua','blue'},
                                'aqua': {'aqua','cyan','turquoise','teal','green','blue'},
                                'cyan': {'cyan','aqua','turquoise','teal','green','blue'},
                                'turquoise': {'turquoise','cyan','aqua','teal','green','blue'},
                                'azure': {'azure','blue','cyan','aqua','turquoise','white'},
                                'blue': {'blue','azure','navy','cyan','aqua','turquoise','indigo','purple'},
                                'navy': {'navy','blue','indigo','purple','violet','black'},
                                'indigo': {'indigo','blue','navy','purple','violet'},
                                'black': {'black','navy','grey','gray','maroon','brown'},
                                'white': {'white','cream','beige','silver','tan'},
                                'gray': {'gray','grey','silver','black','white'},
                                'grey': {'grey','gray','silver','black','white'},
                                'silver': {'silver','grey','gray','white','gold'},
                            }
                            _user_expanded = set()
                            for _uc in _tok_colors:
                                _user_expanded.add(_uc)
                                if _uc in _syn_table:
                                    _user_expanded |= _syn_table[_uc]
                            _obj_in_user = (
                                _obj_color_clean and
                                _obj_color_clean in _user_expanded
                            )
                            if not _obj_in_user:
                                # User said <color> but movable's color (if any) is NOT in syn set.
                                # We treat this as wrong-color → trigger fallback recovery.
                                _wrong_color = True
                                print(f'{bcolors.WARNING}[LMP.py | composer] Color mismatch recovery: user color tok={_tok_colors}, obj.color="{_obj_color_clean}" (expanded user set: {sorted(_user_expanded)[:6]}). Triggering generic fallback to scan variants.{bcolors.ENDC}')
                except Exception as _wc_err:
                    try:
                        import traceback
                        print(f'{bcolors.WARNING}[LMP.py | composer] Wrong-color detection raised (continuing): {_wc_err}{bcolors.ENDC}')
                        traceback.print_exc(limit=1)
                    except Exception:
                        pass
            if _no_action or _wrong_color:
                print(f'{bcolors.WARNING}[LMP.py | {self._name}] No valid execute() call (empty LLM response OR exec error). Applying task-generic fallback action.{bcolors.ENDC}')
                import traceback as _tb_fb
                import re as _re_fb
                import numpy as _np_fb
                import numpy as _np_fb2
                # Make imports + local vars visible to fb_code by injecting into gvars
                gvars = dict(gvars) if gvars else {}
                gvars['_re_fb'] = _re_fb
                gvars['_tb_fb'] = _tb_fb
                gvars['_np_fb'] = _np_fb
                gvars['_np_fb2'] = _np_fb2
                _traceback = _tb_fb
                gvars['_tb2_fb'] = _traceback  # used inside fb_code exception handlers
                gvars['_tb3_fb'] = _traceback
                # `query` is an argument to __call__ but may not be in lvars/gvars for exec_safe,
                # so inject it explicitly:
                gvars['query'] = query
                # Also expose the planner-level instruction 'objects' list (if any) to avoid NameError
                lvars = dict(lvars) if lvars else {}
                fb_code = '\n'.join([
                    '# === FALLBACK: LLM returned empty code; trying generic action heuristics ===',
                    'try:',
                    '    _q_clean = (query or "").strip().lower()',
                    '    # Strip common verbs + articles',
                    '    _q_stripped = _re_fb.sub(r"^(push|press|turn|move|slide|take|pick|place|put|grasp|grab|lift|off|on|open|close|reach|get|set|turn off|switch off|toggle)[ ]+(the |a |an )?", "", _q_clean)',
                    '    _noun = _q_stripped.strip() or "button"',
                    '    print("[LMP fb] noun candidate:", _noun)',
                    '    movable = None',
                    '    # 1a) parse_query_obj on exact noun first, then expand to ALL color-button combos',
                    '    #    (PushButton/LampOff environments have multi-color buttons and parse_query_obj',
                    '    #     often returns None for the exact color phrase, or returns wrong-colored one',
                    '    #     when asked for generic "button"). We exhaust color adjectives.',
                    '    _COMMON_COLORS = [',
                    '        "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple",',
                    '        "violet", "pink", "rose", "brown", "black", "white", "gray", "grey", "navy",',
                    '        "azure", "olive", "teal", "maroon", "silver", "gold", "beige", "tan",',
                    '        "cream", "crimson", "indigo", "lime", "aqua", "coral", "salmon", "turquoise",',
                    '    ]',
                    '    # RLBench color palette is limited; user-facing color words often differ from',
                    '    # internal enum. Use synonym expansion so "purple" instruction -> "violet" enum,',
                    '    # "grey" -> "gray", "turquoise" -> "cyan", "pink" -> "magenta", etc.',
                    '    _COLOR_SYNONYMS = {',
                    '        "purple": ["purple", "violet", "magenta", "pink", "indigo", "crimson"],',
                    '        "violet": ["violet", "purple", "indigo", "magenta"],',
                    '        "magenta": ["magenta", "purple", "pink", "violet", "crimson", "red"],',
                    '        "pink": ["pink", "magenta", "purple", "coral", "salmon", "red", "rose"],',
                    '        "rose": ["rose", "pink", "magenta", "red", "coral", "salmon", "crimson"],',
                    '        "red": ["red", "crimson", "maroon", "magenta", "pink", "coral", "salmon", "rose"],',
                    '        "crimson": ["crimson", "red", "maroon", "magenta", "purple"],',
                    '        "maroon": ["maroon", "red", "crimson", "brown"],',
                    '        "coral": ["coral", "orange", "salmon", "pink", "red"],',
                    '        "salmon": ["salmon", "pink", "coral", "orange", "red"],',
                    '        "orange": ["orange", "coral", "salmon", "gold", "yellow", "brown"],',
                    '        "brown": ["brown", "orange", "maroon", "tan", "beige", "gold"],',
                    '        "tan": ["tan", "brown", "beige", "gold", "orange"],',
                    '        "beige": ["beige", "tan", "cream", "white", "gold", "brown"],',
                    '        "cream": ["cream", "beige", "white", "yellow", "tan"],',
                    '        "yellow": ["yellow", "gold", "orange", "cream", "beige", "lime"],',
                    '        "gold": ["gold", "yellow", "orange", "brown", "tan", "beige"],',
                    '        "olive": ["olive", "green", "yellow", "lime", "brown"],',
                    '        "lime": ["lime", "green", "yellow", "olive", "aqua"],',
                    '        "green": ["green", "lime", "olive", "teal", "aqua", "cyan", "turquoise"],',
                    '        "teal": ["teal", "green", "cyan", "turquoise", "aqua", "blue"],',
                    '        "aqua": ["aqua", "cyan", "turquoise", "teal", "green", "blue"],',
                    '        "cyan": ["cyan", "aqua", "turquoise", "teal", "green", "blue"],',
                    '        "turquoise": ["turquoise", "cyan", "aqua", "teal", "green", "blue"],',
                    '        "azure": ["azure", "blue", "cyan", "aqua", "turquoise", "white"],',
                    '        "blue": ["blue", "azure", "navy", "cyan", "aqua", "turquoise", "indigo", "purple"],',
                    '        "navy": ["navy", "blue", "indigo", "purple", "violet", "black"],',
                    '        "indigo": ["indigo", "blue", "navy", "purple", "violet"],',
                    '        "black": ["black", "navy", "grey", "gray", "maroon", "brown"],',
                    '        "white": ["white", "cream", "beige", "silver", "tan"],',
                    '        "gray": ["gray", "grey", "silver", "black", "white"],',
                    '        "grey": ["grey", "gray", "silver", "black", "white"],',
                    '        "silver": ["silver", "grey", "gray", "white", "gold"],',
                    '    }',
                    '    # Extract any embedded color keyword from noun phrase to prioritize',
                    '    _tok_color = None',
                    '    try:',
                    '        for _c in _COMMON_COLORS:',
                    '            if _c in (_noun or ""):',
                    '                _tok_color = _c',
                    '                break',
                    '    except Exception:',
                    '        pass',
                    '    _tok_color_syns = _COLOR_SYNONYMS.get(_tok_color, [_tok_color]) if _tok_color else None',
                    '    _try_phrases_1a = []',
                    '    if _noun:',
                    '        _try_phrases_1a.append(_noun)',
                    '    if _tok_color_syns is not None:',
                    '        # Put target-color variants FIRST (synonyms!) so correct button is picked earliest',
                    '        for _syn in _tok_color_syns:',
                    '            _try_phrases_1a.append(_syn + " button")',
                    '            _try_phrases_1a.append("the " + _syn + " button")',
                    '            _try_phrases_1a.append(_syn + " switch")',
                    '            _try_phrases_1a.append("the " + _syn + " switch")',
                    '    for _c in _COMMON_COLORS:',
                    '        _try_phrases_1a.append(_c + " button")',
                    '    _try_phrases_1a.extend(["button", "switch", "lever", "handle", "block", "meat", "object"])',
                    '    # Helper: check if a returned movable color matches the target color synonym set',
                    '    def _color_ok(_mv):',
                    '        if _mv is None or _tok_color_syns is None:',
                    '            return True',
                    '        try:',
                    '            _raw = None',
                    '            _raw = getattr(_mv, "color", None)',
                    '            if _raw is None and hasattr(_mv, "get"):',
                    '                try: _raw = _mv.get("color", None)',
                    '                except Exception: _raw = None',
                    '            if callable(_raw):',
                    '                try: _raw = _raw()',
                    '                except Exception: _raw = None',
                    '            _mc = str(_raw or "").strip().lower()',
                    '            if not _mc:',
                    '                return True  # no color info; accept this object',
                    '            # Accept if: mc contains ANY of the target synonyms, OR any target synonym contains mc',
                    '            for _syn in _tok_color_syns:',
                    '                if _syn in _mc or _mc in _syn:',
                    '                    return True',
                    '            return False',
                    '        except Exception:',
                    '            return True',
                    '    for _cand in _try_phrases_1a:',
                    '        if movable is not None: break',
                    '        try:',
                    '            movable = parse_query_obj(_cand)',
                    '            if movable is not None and not _color_ok(movable):',
                    '                movable = None  # Wrong-colored -> keep looking',
                    '            if movable is not None:',
                    '                print("[LMP fb] parse_query_obj got:", _cand)',
                    '                break',
                    '        except Exception:',
                    '            pass',
                    '    # Also try detect() with color synonyms (robust to parser-only failures)',
                    '    if movable is None and _tok_color_syns is not None:',
                    '        for _syn in _tok_color_syns:',
                    '            if movable is not None: break',
                    '            for _det_q in [_syn + " button", _syn, "button", "switch"]:',
                    '                try:',
                    '                    movable = detect(_det_q)',
                    '                    if movable is not None and not _color_ok(movable):',
                    '                        movable = None',
                    '                    if movable is not None:',
                    '                        print("[LMP fb] detect() got:", _det_q)',
                    '                        break',
                    '                except Exception:',
                    '                    pass',
                    '    # 1b) detect() fallback candidates',
                    '    if movable is None:',
                    '        for _dcand in ["button", "switch", "block", "meat", "lever"]:',
                    '            try:',
                    '                movable = detect(_dcand)',
                    '                if movable is not None:',
                    '                    print("[LMP fb] detect() got:", _dcand)',
                    '                    break',
                    '            except Exception:',
                    '                pass',
                    '    # 1c) scan objects in environment (if available)',
                    '    if movable is None:',
                    '        try:',
                    '            _objs = objects if "objects" in dir() else []',
                    '            if isinstance(_objs, list) and len(_objs) > 0:',
                    '                try:',
                    '                    movable = parse_query_obj(_objs[0])',
                    '                    print("[LMP fb] parse_query_obj(objects[0]) =", _objs[0])',
                    '                except Exception:',
                    '                    try:',
                    '                        movable = detect(_objs[0])',
                    '                        print("[LMP fb] detect(objects[0]) =", _objs[0])',
                    '                    except Exception:',
                    '                        movable = None',
                    '        except Exception:',
                    '            pass',
                    '    # 2) Build a SAFE CALLABLE affordance_map lambda (execute requires callable, not value).',
                    '    #    Running get_affordance_map inside the lambda means it executes in the same',
                    '    #    context where value-LMP return semantics (ret_val → lvars) work properly.',
                    '    #    NOTE: All outer-scope variables referenced inside the lambda are passed via',
                    '    #    DEFAULT ARGUMENTS. This avoids Python exec(locals=dict) closure-loss bug:',
                    '    #    functions defined inside an exec with explicit locals cannot close over',
                    '    #    mutable variables stored in the exec-specific locals dict (they end up',
                    '    #    NameError at call-time). Default-arg values are captured at DEF-time.',
                    '    _aff_phrases = []',
                    '    if _tok_color_syns is not None:',
                    '        # Target color-specific phrases FIRST (with SYNONYM expansion! So "purple" user',
                    '        # instruction maps to "violet" RLBench color in get_affordance_map calls.)',
                    '        for _syn in _tok_color_syns:',
                    '            _aff_phrases.extend([',
                    '                "center of the " + _syn + " button",',
                    '                "top of the " + _syn + " button",',
                    '                "center of the " + _syn + " switch",',
                    '                "top of the " + _syn + " switch",',
                    '            ])',
                    '    if _noun:',
                    '        _aff_phrases.extend([',
                    '            "center of the " + _noun,',
                    '            "top of the " + _noun,',
                    '        ])',
                    '    _aff_phrases.extend([',
                    '        "center of the button", "center of the switch", "center of the object",',
                    '        "top of button", "top of switch", "center of block", "center of meat",',
                    '    ])',
                    '    _aff_phrases_local = list(_aff_phrases)  # snapshot value now',
                    '    _np_fb_local = _np_fb2',
                    '    def _fb_aff_lambda(_phrases=_aff_phrases_local, _np=_np_fb_local, _get_empty_aff=get_empty_affordance_map, _get_aff=get_affordance_map):',
                    '        for _p in _phrases:',
                    '            try:',
                    '                _got = _get_aff(_p)',
                    '                try:',
                    '                    _arr = _got.array if hasattr(_got, "array") else _np.asarray(_got)',
                    '                    if _arr.ndim == 3 and float(_arr.max()) > 0.0:',
                    '                        return _got',
                    '                except Exception:',
                    '                    pass',
                    '            except Exception:',
                    '                pass',
                    '        # Fallback: empty map; execute() will use movable center to build fallback affordance.',
                    '        try:',
                    '            return _get_empty_aff()',
                    '        except Exception:',
                    '            return _np.zeros((40, 40, 40), dtype=float)',
                    '    _safe_aff_map = _fb_aff_lambda',
                    '    # 3) Execute. Note: movable may be None (execute() has EE fallback in that case).',
                    '    print("[LMP fb] calling execute(movable=", movable is not None, ")")',
                    '    try:',
                    '        execute(movable, affordance_map=_safe_aff_map)',
                    '        print("[LMP fb] execute OK")',
                    '    except Exception as _efb:',
                    '        print("[LMP fb] execute FAILED:", str(_efb)[:300])',
                    '        try:',
                    '            print(_tb2_fb.format_exc()[-600:])',
                    '        except Exception:',
                    '            pass',
                    'except Exception as _fb_top:',
                    '    print("[LMP fb] top-level error:", str(_fb_top)[:300])',
                    '    try:',
                    '        print(_tb3_fb.format_exc()[-800:])',
                    '    except Exception:',
                    '        pass',
                ])
                try:
                    exec_safe(fb_code, gvars, lvars)
                except Exception as _fbe:
                    print(f'{bcolors.WARNING}[LMP.py | composer] Fallback execution raised: {_fbe}{bcolors.ENDC}')
                    try:
                        print('[LMP.py | composer] Fallback traceback:')
                        print(_tb_fb.format_exc())
                    except Exception:
                        pass

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    # 清理 LLM 生成代码的缩进问题
    lines = code_str.split('\n')
    # 去掉开头空行
    while lines and lines[0].strip() == '':
        lines.pop(0)
    # 去掉结尾空行
    while lines and lines[-1].strip() == '':
        lines.pop()
    if lines:
        # 计算首行缩进，统一 dedent
        first_indent = len(lines[0]) - len(lines[0].lstrip())
        if first_indent > 0:
            lines = [line[first_indent:] if len(line) >= first_indent else line for line in lines]
    # 自动注释非代码行（LLM 有时将查询文本如 "open gripper." 作为代码第一行）
    _py_keywords = {'def', 'for', 'if', 'while', 'return', 'else', 'elif', 'try', 'except', 'finally', 'with', 'import', 'from', 'class', 'pass', 'break', 'continue', 'raise', 'yield', 'global', 'nonlocal', 'assert', 'del', 'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False'}
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            cleaned_lines.append(line)
            continue
        # 检查是否像 Python 代码：包含 ( 或 =，或是 Python 关键字，或是单个标识符
        _first_word = stripped.split()[0].rstrip(':') if stripped.split() else ''
        _looks_like_code = ('(' in stripped or '=' in stripped or ':' in stripped
                            or _first_word in _py_keywords or len(stripped.split()) <= 1)
        if not _looks_like_code:
            cleaned_lines.append('# ' + line)  # 转为注释
        else:
            cleaned_lines.append(line)
    code_str = '\n'.join(cleaned_lines)

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e