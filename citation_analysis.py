#!/usr/bin/env python3
"""
論文で引用されている文献と cites.bib に含まれる文献の比較分析
"""

import re
import os
from pathlib import Path

def extract_cited_keys_from_tex_files(src_dir="src"):
    """srcディレクトリ内のTeXファイルから引用されている文献キーを抽出"""
    cited_keys = set()
    cite_pattern = r'\\cite\{([^}]+)\}'
    
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Warning: {src_dir} directory not found")
        return cited_keys
    
    for tex_file in src_path.glob("*.tex"):
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # \cite{key1,key2,key3} のような複数キーにも対応
            matches = re.findall(cite_pattern, content)
            for match in matches:
                # カンマ区切りの場合を処理
                keys = [key.strip() for key in match.split(',')]
                cited_keys.update(keys)
                
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
    
    return cited_keys

def extract_bib_keys_from_file(bib_file="bib/cites.bib"):
    """cites.bibファイルから全ての文献キーを抽出"""
    bib_keys = set()
    bib_pattern = r'^@[a-zA-Z]+\{([^,]+),'
    
    bib_path = Path(bib_file)
    if not bib_path.exists():
        print(f"Warning: {bib_file} not found")
        return bib_keys
    
    try:
        with open(bib_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(bib_pattern, line.strip())
                if match:
                    bib_keys.add(match.group(1))
                    
    except Exception as e:
        print(f"Error reading {bib_file}: {e}")
    
    return bib_keys

def generate_analysis_report(cited_keys, bib_keys, output_file="citation_analysis_report.md"):
    """分析結果をMarkdownレポートとして生成"""
    
    unused_keys = bib_keys - cited_keys
    cited_but_not_in_bib = cited_keys - bib_keys
    
    report = f"""# 文献引用分析レポート

## 概要
- **cites.bib内の総文献数**: {len(bib_keys)} 件
- **論文本文で引用された文献数**: {len(cited_keys)} 件
- **使用されていない文献数**: {len(unused_keys)} 件
- **bibファイルにない引用**: {len(cited_but_not_in_bib)} 件

## 使用されていない文献 ({len(unused_keys)} 件)
"""
    
    if unused_keys:
        for i, key in enumerate(sorted(unused_keys), 1):
            report += f"{i:2d}. `{key}`\n"
    else:
        report += "すべての文献が使用されています。\n"
    
    if cited_but_not_in_bib:
        report += f"\n## ⚠️ bibファイルにない引用 ({len(cited_but_not_in_bib)} 件)\n"
        for i, key in enumerate(sorted(cited_but_not_in_bib), 1):
            report += f"{i:2d}. `{key}`\n"
    
    report += f"\n## 引用されている文献一覧 ({len(cited_keys)} 件)\n"
    for i, key in enumerate(sorted(cited_keys), 1):
        report += f"{i:2d}. `{key}`\n"
    
    # レポートをファイルに保存
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分析レポートを {output_file} に保存しました")
    except Exception as e:
        print(f"Error writing report: {e}")
    
    return report

def main():
    print("文献引用分析を開始します...")
    
    # 引用されている文献キーを抽出
    print("論文本文から引用キーを抽出中...")
    cited_keys = extract_cited_keys_from_tex_files()
    print(f"引用されている文献: {len(cited_keys)} 件")
    
    # bibファイルから全文献キーを抽出
    print("cites.bibから文献キーを抽出中...")
    bib_keys = extract_bib_keys_from_file()
    print(f"bibファイル内の文献: {len(bib_keys)} 件")
    
    # 分析レポート生成
    print("分析レポートを生成中...")
    report = generate_analysis_report(cited_keys, bib_keys)
    
    # 簡単な統計をコンソールに表示
    unused_count = len(bib_keys - cited_keys)
    print(f"\n=== 分析結果 ===")
    print(f"使用されていない文献: {unused_count} 件")
    print(f"使用率: {len(cited_keys)/len(bib_keys)*100:.1f}%")

if __name__ == "__main__":
    main() 