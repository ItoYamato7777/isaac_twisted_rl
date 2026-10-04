# Claude Code の仕様の確認 (Phase 0-3)

Phase 2 で Claude 基盤を設計するときに読む。設計の前提にした仕様と、その出典をまとめる。

- 確認した日: 2026-10-03
- 対象: Claude Code 2.1.287 (VS Code 拡張の native binary、`--setting-sources=user,project,local --permission-mode auto` で起動)
- 方法: 公式ドキュメントを自分で WebFetch で開き、該当箇所を原文のまま引用させて確かめた。
  claude-code-guide subagent にも同じ項目を調べさせ、結果を突き合わせた。
- 印の意味
  - **[確認]** 公式ドキュメントの原文で確かめた
  - **[未確認]** ドキュメントに記述がない、または自分では確かめていない

出典 (すべて 2026-10-03 に取得):

| 略号 | URL |
|---|---|
| MEM | https://code.claude.com/docs/en/memory |
| SUB | https://code.claude.com/docs/en/sub-agents |
| SKL | https://code.claude.com/docs/en/skills |
| HK | https://code.claude.com/docs/en/hooks |
| SET | https://code.claude.com/docs/en/settings |
| PRM | https://code.claude.com/docs/en/permissions |
| WT | https://code.claude.com/docs/en/worktrees |

---

## 1. CLAUDE.md

| 項目 | 仕様 | 出典 |
|---|---|---|
| 起動時に読むもの | 作業ディレクトリとその上の全ディレクトリの CLAUDE.md・CLAUDE.local.md。すべて連結され、上書きはされない。順番はルートから作業ディレクトリへ。同じディレクトリでは CLAUDE.local.md が後 | [確認] MEM「How CLAUDE.md files load」 |
| サブディレクトリの CLAUDE.md | 起動時には読まない。Claude がそのサブディレクトリのファイルを読んだときに読み込まれる | [確認] MEM 同上 |
| `@path` の import | 起動時に展開され、毎回全文が文脈に入る。「import は整理には役立つが、文脈の消費は減らさない」と明記。相対パスは import を書いたファイルからの相対。再帰は最大 4 段 | [確認] MEM「Import additional files」「Write effective instructions」 |
| import の例外 | コードスパンと囲みコードブロックの中の `@` は import されない。パスを書くだけなら `` `@README` `` のように backtick で囲む | [確認] MEM 同上 |
| 作業ディレクトリの外の import | 初回に承認のダイアログが出る。断ると無効のまま | [確認] MEM 同上 |
| 大きさの目安 | 1 ファイル 200 行未満。長いほど文脈を使い、守られにくくなる。一部のコードにだけ要る指示は、パスを限定した rules へ移す | [確認] MEM「Write effective instructions」 |
| HTML コメント | ブロックの `<!-- -->` は文脈に入る前に取り除かれる (コードブロックの中は残る)。人間向けの注記に使える | [確認] MEM「How CLAUDE.md files load」 |
| 読み込みの除外 | `claudeMdExcludes` (glob、絶対パス)。全階層の設定に書け、配列は合算される | [確認] MEM「Exclude specific CLAUDE.md files」 |
| 位置づけ | CLAUDE.md と auto memory は「文脈であって、強制される設定ではない」。必ず止めたい操作は PreToolUse hook を使う | [確認] MEM「CLAUDE.md vs auto memory」 |

## 2. `.claude/rules/`

| 項目 | 仕様 | 出典 |
|---|---|---|
| 置き方 | `.claude/rules/` の下の `.md`。サブディレクトリも再帰的に探す | [確認] MEM「Organize rules with .claude/rules/」 |
| `paths` のない rule | 起動時に、`.claude/CLAUDE.md` と同じ優先度で読み込まれる | [確認] MEM 同上 |
| `paths` のある rule | frontmatter の `paths` (glob のリスト) に合うファイルを Claude が扱うときだけ適用される | [確認] MEM「Path-specific rules」 |
| ユーザーの rule | `~/.claude/rules/` は全プロジェクトに効く | [確認] MEM「User-level rules」 |
| `paths` 以外の frontmatter | claude-code-guide は「`paths` 以外は無視される」と報告した | [未確認] |

## 3. Skills

| 項目 | 仕様 | 出典 |
|---|---|---|
| 置き場所 | `.claude/skills/<name>/SKILL.md`。`.claude/commands/<name>.md` も同じ `/name` を作り、今も動く (古い形式。`name`・`paths` は使えない) | [確認] SKL |
| 文脈に常にあるもの | skill の名前と description の一覧。本文は、ユーザーが `/name` で呼んだときか、Claude が自動で呼んだときに 1 つのメッセージとして入り、以後のターンにも残る。skill のファイルは後のターンで読み直されない | [確認] SKL |
| description | Claude はこれを見て使うかどうかを決める。省略すると本文の最初の空でない行が使われる。`description` と `when_to_use` を合わせて 1,536 文字で切られる。skill が多いと一覧の文字数の予算 (モデルの文脈の 1%) に収めるため、一部の description が落ちる。→ 用途を先頭に書く | [確認] SKL |
| 主な frontmatter | `name`、`description`、`when_to_use`、`argument-hint`、`arguments`、`disable-model-invocation` (true で Claude が自動で呼ばない。一覧にも載らない)、`user-invocable` (false で `/` メニューから隠す)、`allowed-tools`、`disallowed-tools`、`model`、`effort`、`context: fork`、`agent`、`background`、`hooks`、`paths` (合うファイルを扱うときだけ自動で読み込む)、`shell`、`metadata` | [確認] SKL |
| 編集の反映 | `.claude/skills/` の変更はその session の中で反映される (bare mode を除く)。session の開始時に無かった最上位の skills ディレクトリを作ったときは `/reload-skills` が要る | [確認] SKL |
| subagent からの利用 | subagent は Skill ツールでプロジェクト・ユーザー・プラグインの skill を呼べる。agent 定義の `skills:` に書いたものは起動時に全文が入る。`disable-model-invocation: true` の skill は事前に読み込めない | [確認] SUB「Preloading skills into subagents」 |

## 4. Subagents

| 項目 | 仕様 | 出典 |
|---|---|---|
| 定義 | `.claude/agents/<name>.md`。必須は `name` と `description`。認識されない項目は、エラーを出さずに無視される | [確認] SUB「Supported frontmatter fields」 |
| ツールの制限 | `tools` (許可のリスト。省略するとすべて継承)、`disallowedTools` (除くもの) | [確認] SUB 同上 |
| モデル | `model`: `sonnet`・`opus`・`haiku`・`fable`・完全なモデル ID・`inherit` | [確認] SUB 同上 |
| その他の項目 | `permissionMode`、`maxTurns`、`skills`、`mcpServers`、`hooks`、`memory`、`background`、`omitClaudeMd` (2.1.271 以降)、`effort`、`isolation: worktree`、`color`、`initialPrompt` | [確認] SUB 同上 |
| 起動時の文脈 | subagent 自身のシステムプロンプト、委任のメッセージ、**メインの会話が読み込む CLAUDE.md の全階層** (`~/.claude/CLAUDE.md`、project rules、CLAUDE.local.md を含む)、**git status のスナップショット**、`skills:` で指定した skill の全文 | [確認] SUB「What loads at startup」 |
| 例外 | 組み込みの Explore と Plan は CLAUDE.md と git status を読まない | [確認] SUB 同上 |
| 渡らないもの | メインの会話の履歴、メインが読んだファイル、output style、**メインの auto memory** (fork の場合だけ会話を引き継ぐ) | [確認] SUB「State that never reaches a non-fork subagent」 |

## 5. Hooks

| 項目 | 仕様 | 出典 |
|---|---|---|
| イベント | SessionStart、Setup、UserPromptSubmit、UserPromptExpansion、PreToolUse、PermissionRequest、PermissionDenied (auto mode が拒否したとき)、PostToolUse、PostToolUseFailure、PostToolBatch、Notification、MessageDisplay、SubagentStart、SubagentStop、TaskCreated、TaskCompleted、Stop、StopFailure、TeammateIdle、InstructionsLoaded (CLAUDE.md・rules が読み込まれたとき)、ConfigChange、CwdChanged、DirectoryAdded、FileChanged、WorktreeCreate、WorktreeRemove、PreCompact、PostCompact、PreModelSwitch、PostModelSwitch、Elicitation、ElicitationResult、SessionEnd | [確認] HK のイベント一覧 |
| 共通の入力 (stdin の JSON) | `session_id`、`prompt_id`、`transcript_path`、`cwd`、`scratchpad_dir`、`permission_mode`、`effort`、`hook_event_name`、(subagent の中では) `agent_id`・`agent_type` | [確認] HK「Common input fields」 |
| PreToolUse の入力 | `tool_name`、`tool_input`、`tool_use_id`。Edit/Write は `tool_input.file_path`、Bash は `tool_input.command` | [確認] HK |
| PostToolUse の入力 | 上に加えて `tool_response`。ツールの実行は止められない。`decision: "block"` と `reason`、または `additionalContext` で Claude にフィードバックする | [確認] HK |
| SessionStart の入力 | `source`: `startup`・`resume`・`clear`・`compact`・`fork`。プレーンテキストの stdout はそのまま文脈に入る (JSON なら `hookSpecificOutput.additionalContext`) | [確認] HK |
| 終了コード 0 | 成功。多くのイベントで stdout はデバッグログへ行くだけ。UserPromptSubmit・UserPromptExpansion・SessionStart・PostModelSwitch では stdout が文脈に入る | [確認] HK「Exit code 0」 |
| 終了コード 2 | 止められるイベントでは操作を止める (PreToolUse ならツールの呼び出しを止める)。JSON の `permissionDecision: "allow"` でも覆らない。理由は JSON の reason、なければ stderr | [確認] HK「Exit code 2」 |
| **終了コード 1 など** | **止めない** (エラーの通知が出て、操作は進む)。「方針を強制する hook は `exit 2` を使え」と明記 | [確認] HK |
| JSON での制御 | PreToolUse は `hookSpecificOutput.permissionDecision` (`allow`・`deny`・`ask`・`defer`) と `permissionDecisionReason` | [確認] HK |
| matcher | `"*"`・空・省略はすべて。英数字と `_` `-` だけなら完全一致。`Edit\|Write` や `Edit, Write` は列挙。それ以外の記号を含むと JavaScript の正規表現。SessionStart の matcher は `startup\|resume` など | [確認] HK |
| 設定の書式 | `hooks` → イベント名 → `[{matcher, hooks: [{type: "command", command, timeout, ...}]}]`。handler の型は `command`・`http`・`mcp_tool`・`prompt`・`agent`。`if` に permission の規則を書いて絞れる (例 `Bash(rm *)`) | [確認] HK |
| 環境変数 | `${CLAUDE_PROJECT_DIR}` は session を始めたプロジェクトのルート (worktree に入っても変わらない。worktree のパスは入力の `cwd`) | [確認] HK、WT |
| 既定の timeout | command・http・mcp_tool は 600 秒。UserPromptSubmit などは 30 秒。SessionEnd は合計 1.5 秒 (最大 60 秒まで延ばせる) | [確認] HK |
| 設定の変更の反映 | 「設定ファイルの hooks を直接編集すると、通常はファイルの監視で自動的に反映される」。`/hooks` は読み取り専用の一覧 | [確認] HK |

## 6. 設定ファイル

| 項目 | 仕様 | 出典 |
|---|---|---|
| 優先順位 (高い順) | managed → コマンドライン (`--settings`) → `.claude/settings.local.json` → `.claude/settings.json` → `~/.claude/settings.json` | [確認] SET「Settings precedence」 |
| 役割 | `settings.json` はリポジトリにコミットして全員で共有する。`settings.local.json` は個人用で、Claude Code が最初に書くときにグローバルな git の除外に `**/.claude/settings.local.json` を加える (手で作ったときは自分で除外する) | [確認] SET |
| リストの合算 | `permissions.allow` などのリストは、上書きではなく全ファイルの分が合算される | [確認] SET「Lists merge instead of overriding」 |
| 読む場所 | 共有の `.claude/settings.json` は session の主作業ディレクトリから読む (親へは遡らない)。`settings.local.json` は git のルートから読む | [確認] SET、PRM |
| 信頼の確認 | コミットされた設定の `allow` と `additionalDirectories` は、フォルダを信頼するまで効かない。`deny` と `ask` はすぐ効く | [確認] SET |

## 7. Permissions

| 項目 | 仕様 | 出典 |
|---|---|---|
| 範囲をまたぐ優先 | どの範囲の deny も、ほかの範囲の allow に勝つ。「ユーザー設定が allow、プロジェクト設定が deny なら deny が止める」と明記 | [確認] PRM「Settings precedence」 |
| Bash の `*` | 空白を含む任意の文字列に合う。最初の `*` より前は書いたとおりに比べる。`Bash(ls *)` は `ls` と `ls -la` に合い `lsof` に合わない。`:*` は末尾に限り ` *` と同じ | [確認] PRM「Wildcard patterns」 |
| 複合コマンド | `&&`・`\|\|`・`;`・`\|`・`&`・改行で分け、allow はすべての部分に合う必要がある。deny と ask は、どれか 1 つの部分 (サブシェル・コマンド置換・for の本体も含む) に合えば効く。auto mode でも効く | [確認] PRM「Compound commands」 |
| 剥がされる包み | `timeout`・`time`・`nice`・`nohup`・`stdbuf`・`command`・`builtin`・フラグなしの `xargs` は、照合の前に剥がされる。安全と分かっている環境変数の代入も剥がされる (deny と ask は、どんな代入も越えて照合する) | [確認] PRM「Wrappers」 |
| Bash の規則の限界 | 同じプログラムでも別の書き方 (`/bin/rm`、`bash -c '...'`、`git -C . push`) には合わない。「安全の境界ではない」と明記 | [確認] PRM「What a Bash rule doesn't match」 |
| Read・Edit のパス | gitignore の書式。`//path` は絶対パス、`~/path` はホームから、`/path` は設定ファイルの基点から (プロジェクト設定では主作業ディレクトリ)、`path` と `./path` は現在のディレクトリから。`/Users/x` は絶対パスではない | [確認] PRM「Read and Edit」 |
| Read・Edit の deny が効く範囲 | Claude の組み込みのファイルツールに加え、Bash で Claude Code が認識するファイルのコマンド (`cat`・`head`・`tail`・`sed`・`tee`) と、リダイレクトの先 (`> file`、`< file`) にも効く。ファイル名を書かずに読むコマンド (`grep -r pattern .`) や、Python のスクリプトのようにファイルを自分で開く子プロセスには効かない | [確認] PRM の Warning |
| Read の deny | 同じパスへの Edit と Write も止める (NotebookEdit は止めない) | [確認] PRM |
| 規則が使われないツール名 | `Write(...)`・`Glob(...)` などのパスの規則は受け付けるが参照されない。`Edit(...)`・`Read(...)` を使う | [確認] PRM |
| 追加の作業ディレクトリ | `--add-dir` (起動時)・`/add-dir` (session 中)・設定の `additionalDirectories` (永続)。追加したディレクトリは確認なしで読める。設定の `additionalDirectories` で足したものは、ファイルの読み書きの権限を与えるだけで、そこの `.claude/` の設定は読まない | [確認] PRM「Working directories」 |
| `/add-dir` の永続化 | claude-code-guide は「既定で `settings.local.json` に永続化される」と報告したが、ドキュメントでは確かめられなかった | [未確認] |
| 作業ディレクトリの外を読まない設定 | `permissions.blockReadsOutsideWorkingDirectories` | [確認] PRM (設定の詳細は未確認) |
| auto mode | 人の代わりに分類器が操作を審査する。deny と ask の規則はそのまま効く。分類器が何を通すかの詳細は確かめていない | 一部 [確認] PRM。詳細は [未確認] |

## 8. Auto memory

| 項目 | 仕様 | 出典 |
|---|---|---|
| 場所 | `~/.claude/projects/<project>/memory/`。`<project>` は git リポジトリから決まり、worktree とサブディレクトリで共有される | [確認] MEM「Storage location」 |
| 起動時に読むもの | `MEMORY.md` の先頭 200 行か 25 KB の少ない方。個別のファイルは必要なときに Claude が読む | [確認] MEM「How it works」 |
| 止め方 | 設定の `autoMemoryEnabled: false` (プロジェクトの設定に書けば、そのプロジェクトだけ)、または環境変数 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` | [確認] MEM「Enable or disable auto memory」 |
| subagent | メインの auto memory は渡らない (fork を除く) | [確認] SUB |

## 9. Worktree (`isolation: worktree`)

| 項目 | 仕様 | 出典 |
|---|---|---|
| 基点 | 既定 (`worktree.baseRef: "fresh"`) はリモートの既定のブランチ (通常は `origin/main`)。`"head"` にすると現在のローカルの HEAD から切る。ブランチ名は指定できない | [確認] WT「Choose the base branch」 |
| 中身 | git 管理下のファイルだけの新しい checkout。未追跡と gitignore のファイルは入らない (`.worktreeinclude` に書いた gitignore 対象のファイルだけは写される) | [確認] WT |
| 置き場所 | `.claude/worktrees/<name>/` | [確認] WT |
| CLAUDE.md | worktree の中の subagent は、指示のファイルをメインの会話から受け取る。`.claude/worktrees/` の下の worktree のルートにある CLAUDE.md と `.claude/rules/` は読まない | [確認] WT「Isolate subagents with worktrees」 |
| 後始末 | 変更がなければ subagent の終了時に自動で消える | [確認] WT |

---

## 10. claude-code-guide の報告との違い

自分で開いたドキュメントと食い違った点。この 2 点は誤りとして扱う。

| 報告の記述 | ドキュメント |
|---|---|
| 優先順位は「managed > user > project > local」 | managed → コマンドライン → local → project → user (SET) |
| `/add-dir` は既定で `settings.local.json` に永続化される | 記述が見つからない。永続化の手段として書かれているのは `additionalDirectories` だけ (PRM) |

ほかに、報告にあって自分では確かめていない記述 (4 MiB を超える CLAUDE.md は読まれない、`/doctor prompt-audit`、subagent のモデルの決まり方の順序、auto mode の分類器が通すもの) は、この文書の根拠には使わない。

---

## 11. 設計への含意 (Phase 2 で使う)

1. **大きな文書を CLAUDE.md から `@` import しない。** import は毎回全文が文脈に入る (§1)。導線はパスを書くだけにし、`@` を backtick で囲む。
2. **局所の規則は、サブディレクトリの CLAUDE.md か `paths` 付きの rules に置く。** どちらも、そのファイルを扱うときだけ読み込まれる (§1、§2)。
   例: `common/topology/` の import の禁止、`versions/` の追加の規約。
3. **規則を強制する hook は `exit 2` か JSON の `permissionDecision: "deny"` で止める。** `exit 1` では止まらない (§5)。
4. **`logs/`・`outputs/` を守る Edit の deny は、学習スクリプトの書き込みを妨げない。** deny は Python の子プロセスには効かない (§7)。
   Claude の Edit・Write、Bash の `sed`・`tee`・リダイレクトは止まる。
5. **旧実装のディレクトリを読ませないために、プロジェクトの `settings.json` に Read の deny を置ける。** ユーザー設定の allow `Read(//home/itoyama/work/IsaacLab/**)` に勝つ (§7)。
   ただし `grep -r` と Python のスクリプトには効かない。また、IsaacLab 本体のソースを参照する正当な読み込みまで止めないよう、knot_tying の下だけに限る必要がある。
6. **hook と skill は作業中の session にも反映される** (§3、§5)。ただし、まだ無い `.claude/skills/` を新しく作ったときは `/reload-skills` が要る。

### cold-start テストの条件への含意

1. **回答役 (general-purpose) は、CLAUDE.md の全階層と git status を受け取り、auto memory は受け取らない** (§4、§8)。
   - Phase 1 の基準値: CLAUDE.md はどこにもない (`~/.claude/CLAUDE.md` もない)。
   - Phase 5: CLAUDE.md が自動で入る。その行数を読んだ量に足す。
2. **CLAUDE.md はメインの会話が読み込んだものが渡る** (§4、§9)。
   そのため Phase 5 の測定は、CLAUDE.md をコミットした後に**新しい session で**行う。
   回答役には、自分の文脈に CLAUDE.md があるかどうかを申告させて確かめる。
3. **回答役に `refactor/` と正答を見せない方法として、`isolation: worktree` が使える** (§9)。
   worktree には git 管理下のファイルしか入らない。そのため `refactor/` を git の管理外 (`.git/info/exclude`) にしておけば、回答役の checkout に `refactor/` は存在しない。
   - Phase 1: 既定の `fresh` は `origin/main` から切られる。今は 6f41ef1 で、測りたい状態と一致する。
     注意: このリポジトリでは `refs/remotes/origin/HEAD` が手元に記録されていない (`git symbolic-ref` が失敗する)。
     その場合 Claude Code は 5 秒まで取得を試み、だめならローカルの HEAD を使う (WT)。
     Phase 1 の間は作業ブランチに `refactor/` の外のコミットをしないので、どちらでも 6f41ef1 になる。
   - Phase 5: `worktree.baseRef: "head"` を設定するか、`git worktree add` で手で作る。
4. auto memory は回答役に渡らない。そのため、このプロジェクトの memory の中身は測定に影響しない。
   memory のフォルダは 2026-10-03 15:45 の時点で空。

---

## 12. 観察: ドキュメントと食い違った挙動 (2026-10-03、Phase 1 の G-3)

- **auto memory が subagent に入っていた。** cold-start テストの回答役 2 (general-purpose、`isolation: worktree`) は、「CLAUDE.md の指示は文脈に無く、auto memory の `MEMORY.md` の 1 行だけが入っていた」と申告した。
  - spec §4・§8 (SUB「State that never reaches a non-fork subagent」) では、メインの auto memory は subagent に渡らないはず。
  - 申告だけで、トランスクリプトでは確かめていない。そのため [未確認] とする。
  - 当時の MEMORY.md の中身は session の区切り方の好み 1 行だけで、質問とは無関係。今回の測定への影響はない。
  - Phase 5 の測定の前にも、MEMORY.md にリポジトリの事実が入っていないことを確かめる。
- **worktree の中の git。** worktree の `.git` はファイルで、実体はメインの checkout の `.git/worktrees/` にある。
  回答役 2 は「読んではいけない別の checkout の中にある」と判断し、git の履歴を使わなかった。回答役 1 が使ったかは、回答のファイルで確かめる。
