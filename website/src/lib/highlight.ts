export interface HighlightToken {
  text: string;
  type: string;
}

const COLORS: Record<string, string> = {
  keyword: 'text-ctp-mauve',
  string: 'text-ctp-green',
  comment: 'text-ctp-overlay0',
  number: 'text-ctp-peach',
  function: 'text-ctp-blue',
  type: 'text-ctp-yellow',
  operator: 'text-ctp-sky',
  punctuation: 'text-ctp-overlay2',
  flag: 'text-ctp-teal',
  variable: 'text-ctp-red',
  plain: 'text-ctp-text',
};

export function tokenColor(type: string): string {
  return COLORS[type] ?? COLORS.plain;
}

interface Rule {
  type: string;
  pattern: RegExp;
}

const BASH_RULES: Rule[] = [
  { type: 'comment', pattern: /#[^\n]*/ },
  { type: 'string', pattern: /"(?:[^"\\]|\\.)*"|'[^']*'/ },
  { type: 'variable', pattern: /\$\w+|\$\{[^}]+\}/ },
  { type: 'flag', pattern: /--?\w[\w-]*=?/ },
  {
    type: 'keyword',
    pattern:
      /\b(?:if|then|else|elif|fi|for|while|do|done|case|esac|function|return|export|source|cd|mkdir|sudo|apt|npm|cmake|pip|python|xvfb-run|bash)\b/,
  },
  { type: 'number', pattern: /\b\d+(?:\.\d+)?\b/ },
  { type: 'operator', pattern: /[|&;><]|\|\||&&/ },
];

const C_RULES: Rule[] = [
  { type: 'comment', pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\// },
  { type: 'string', pattern: /"(?:[^"\\]|\\.)*"/ },
  {
    type: 'keyword',
    pattern:
      /\b(?:if|else|for|while|do|return|break|continue|switch|case|default|struct|typedef|enum|const|static|extern|inline|void|int|float|double|char|unsigned|signed|long|short|bool|true|false|NULL|sizeof)\b/,
  },
  {
    type: 'type',
    pattern:
      /\b(?:vec[234]|ivec[234]|uvec[234]|mat[234]|sampler\w+|layout|uniform|buffer|in|out|flat|readonly|writeonly|coherent|volatile|restrict)\b/,
  },
  {
    type: 'keyword',
    pattern:
      /\b(?:gl_GlobalInvocationID|gl_LocalInvocationID|gl_WorkGroupID|gl_NumWorkGroups|local_size_x|local_size_y|local_size_z)\b/,
  },
  {
    type: 'function',
    pattern:
      /\b(?:floor|ceil|clamp|mix|step|smoothstep|length|normalize|dot|cross|abs|sign|min|max|sqrt|pow|exp|log|sin|cos|tan|asin|acos|atan|fract|mod|texture|imageLoad|imageStore|barrier|memoryBarrier|atomicAdd)\b/,
  },
  { type: 'number', pattern: /\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fuUlL]?\b/ },
  {
    type: 'keyword',
    pattern: /^#\s*(?:version|define|ifdef|ifndef|endif|include|pragma)\b/m,
  },
  { type: 'operator', pattern: /[+\-*/%=!<>&|^~?:]|->|<<|>>/ },
  { type: 'punctuation', pattern: /[{}()[\];,.]/ },
];

const PYTHON_RULES: Rule[] = [
  { type: 'comment', pattern: /#[^\n]*/ },
  {
    type: 'string',
    pattern:
      /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/,
  },
  {
    type: 'keyword',
    pattern:
      /\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield|True|False|None)\b/,
  },
  {
    type: 'function',
    pattern: /\b(?:print|len|range|enumerate|zip|map|filter|sorted|type|int|float|str|list|dict|set|tuple|open|super|isinstance|hasattr|getattr|setattr)\b/,
  },
  { type: 'number', pattern: /\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/ },
  { type: 'operator', pattern: /[+\-*/%=!<>&|^~@:]|->|<<|>>|\*\*/ },
  { type: 'punctuation', pattern: /[{}()[\];,.]/ },
];

const TS_RULES: Rule[] = [
  { type: 'comment', pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\// },
  {
    type: 'string',
    pattern: /`(?:[^`\\]|\\.)*`|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/,
  },
  {
    type: 'keyword',
    pattern:
      /\b(?:abstract|as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|module|namespace|new|of|package|private|protected|public|readonly|return|set|static|super|switch|this|throw|try|type|typeof|var|void|while|with|yield|true|false|null|undefined)\b/,
  },
  {
    type: 'type',
    pattern:
      /\b(?:string|number|boolean|object|symbol|bigint|any|never|unknown|void|Promise|Array|Record|Partial|Required|Pick|Omit|Map|Set)\b/,
  },
  { type: 'number', pattern: /\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/ },
  { type: 'operator', pattern: /[+\-*/%=!<>&|^~?:]|=>|\.{3}|\?\?/ },
  { type: 'punctuation', pattern: /[{}()[\];,.]/ },
];

const JSON_RULES: Rule[] = [
  { type: 'string', pattern: /"(?:[^"\\]|\\.)*"/ },
  { type: 'keyword', pattern: /\b(?:true|false|null)\b/ },
  { type: 'number', pattern: /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/ },
  { type: 'punctuation', pattern: /[{}[\]:,]/ },
];

const LANG_RULES: Record<string, Rule[]> = {
  bash: BASH_RULES,
  sh: BASH_RULES,
  terminal: BASH_RULES,
  c: C_RULES,
  glsl: C_RULES,
  python: PYTHON_RULES,
  py: PYTHON_RULES,
  typescript: TS_RULES,
  ts: TS_RULES,
  javascript: TS_RULES,
  js: TS_RULES,
  json: JSON_RULES,
};

export function highlight(code: string, lang?: string): HighlightToken[] {
  try {
    const rules = lang ? LANG_RULES[lang.toLowerCase()] : undefined;
    if (!rules) {
      return [{ text: code, type: 'plain' }];
    }

    const tokens: HighlightToken[] = [];
    let pos = 0;

    while (pos < code.length) {
      let best: { match: RegExpExecArray; type: string } | null = null;

      for (const rule of rules) {
        rule.pattern.lastIndex = 0;
        const re = new RegExp(rule.pattern.source, rule.pattern.flags);
        re.lastIndex = pos;
        const m = re.exec(code);
        if (m && m.index === pos) {
          if (!best || m[0].length > best.match[0].length) {
            best = { match: m, type: rule.type };
          }
        }
      }

      if (best && best.match[0].length > 0) {
        tokens.push({ text: best.match[0], type: best.type });
        pos += best.match[0].length;
      } else {
        // Accumulate plain text
        const last = tokens[tokens.length - 1];
        if (last && last.type === 'plain') {
          last.text += code[pos];
        } else {
          tokens.push({ text: code[pos], type: 'plain' });
        }
        pos++;
      }
    }

    return tokens;
  } catch {
    return [{ text: code, type: 'plain' }];
  }
}
