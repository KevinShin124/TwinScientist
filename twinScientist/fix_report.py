import sys; sys.stdout.reconfigure(encoding='utf-8')

with open('core/nodes.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = """parts.append(rationale_abstract.strip())
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 三、技术手段（Technical Details）"""

new = """ra = rationale_abstract
    for header in ["### 支撑事实（来自文献调研）", "## 二、解决思路（Rationale）", "## 六、摘要（Paper Abstract）"]:
        ra = ra.replace(header, "")
    parts.append(ra.strip())
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## 三、技术手段（Technical Details）"""

if old in content:
    print("Replacing...")
    content = content.replace(old, new)
    with open('core/nodes.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done.")
else:
    print("Pattern not found. Searching for rationale_abstract.strip()...")
    idx = content.find("rationale_abstract.strip()")
    if idx >= 0:
        snippet = content[idx:idx+250]
        print(repr(snippet))
    else:
        print("NOT FOUND at all!")
