#!/usr/bin/env python3
"""
論文で引用されている文献と cites.bib に含まれる文献の比較分析
"""

import re
import os
import shutil
from datetime import datetime
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

def extract_cited_keys_in_order(src_dir="src"):
    """srcディレクトリ内のTeXファイルから引用を登場順に抽出"""
    cited_keys_ordered = []
    seen_keys = set()
    cite_pattern = r'\\cite\{([^}]+)\}'
    
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Warning: {src_dir} directory not found")
        return cited_keys_ordered
    
    # ファイル名順でソートして処理順序を安定化
    tex_files = sorted(src_path.glob("*.tex"))
    
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # \cite{key1,key2,key3} のような複数キーにも対応
            matches = re.findall(cite_pattern, content)
            for match in matches:
                # カンマ区切りの場合を処理
                keys = [key.strip() for key in match.split(',')]
                for key in keys:
                    if key not in seen_keys:
                        cited_keys_ordered.append(key)
                        seen_keys.add(key)
                        
        except Exception as e:
            print(f"Error reading {tex_file}: {e}")
    
    return cited_keys_ordered

def parse_bib_entries(bib_file="bib/cites.bib"):
    """bibファイルを解析してエントリごとに分割"""
    bib_entries = {}
    current_entry = []
    current_key = None
    entry_pattern = r'^@[a-zA-Z]+\{([^,]+),'
    
    bib_path = Path(bib_file)
    if not bib_path.exists():
        print(f"Warning: {bib_file} not found")
        return bib_entries
    
    try:
        with open(bib_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            # 新しいエントリの開始をチェック
            match = re.match(entry_pattern, line.strip())
            if match:
                # 前のエントリを保存
                if current_key and current_entry:
                    bib_entries[current_key] = ''.join(current_entry)
                
                # 新しいエントリを開始
                current_key = match.group(1)
                current_entry = [line]
            else:
                # 既存のエントリに行を追加
                if current_entry is not None:
                    current_entry.append(line)
        
        # 最後のエントリを保存
        if current_key and current_entry:
            bib_entries[current_key] = ''.join(current_entry)
            
    except Exception as e:
        print(f"Error reading {bib_file}: {e}")
    
    return bib_entries

def create_backup(file_path):
    """ファイルのバックアップを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"バックアップを作成しました: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"バックアップ作成エラー: {e}")
        return None

def reorder_bibliography_by_citation_order(src_dir="src", bib_file="bib/cites.bib", output_file=None):
    """引用順に従ってbibliographyファイルを並べ替え"""
    if output_file is None:
        output_file = bib_file
    
    # バックアップを作成
    if output_file == bib_file:
        backup_path = create_backup(bib_file)
        if not backup_path:
            print("バックアップ作成に失敗しました。処理を中止します。")
            return None, None
    
    print("本文から引用順序を抽出中...")
    cited_keys_ordered = extract_cited_keys_in_order(src_dir)
    print(f"引用された文献（順序付き）: {len(cited_keys_ordered)} 件")
    
    print("bibファイルを解析中...")
    bib_entries = parse_bib_entries(bib_file)
    print(f"bibファイル内の文献: {len(bib_entries)} 件")
    
    # 引用順にエントリを並べ替え
    reordered_entries = []
    used_keys = set()
    
    # まず引用されている文献を引用順に追加
    for key in cited_keys_ordered:
        if key in bib_entries:
            reordered_entries.append(bib_entries[key])
            used_keys.add(key)
        else:
            print(f"Warning: 引用されているがbibファイルにない文献: {key}")
    
    # 引用されていない文献を最後に追加
    unused_entries = []
    for key, entry in bib_entries.items():
        if key not in used_keys:
            unused_entries.append(entry)
    
    # ファイルに書き込み
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 引用されている文献を書き込み
            for entry in reordered_entries:
                f.write(entry)
                if not entry.endswith('\n\n'):
                    f.write('\n')
            
            # 引用されていない文献があれば追加
            if unused_entries:
                f.write('\n% ========== 以下は引用されていない文献 ==========\n\n')
                for entry in unused_entries:
                    f.write(entry)
                    if not entry.endswith('\n\n'):
                        f.write('\n')
        
        print(f"並べ替えられたbibファイルを {output_file} に保存しました")
        print(f"引用順文献: {len(reordered_entries)} 件")
        print(f"未引用文献: {len(unused_entries)} 件")
        
        return cited_keys_ordered, len(unused_entries)
        
    except Exception as e:
        print(f"Error writing reordered bibliography: {e}")
        return None, None

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
    
    # 引用順並べ替え機能を実行
    print(f"\n=== 引用順並べ替え ===")
    cited_order, unused_entries_count = reorder_bibliography_by_citation_order()
    
    if cited_order:
        print(f"引用順序:")
        for i, key in enumerate(cited_order[:10], 1):  # 最初の10件を表示
            print(f"  {i:2d}. {key}")
        if len(cited_order) > 10:
            print(f"  ... （他 {len(cited_order) - 10} 件）")

def reorder_only():
    """引用順並べ替えのみを実行する関数"""
    print("cites.bibを引用順に並べ替えます...")
    cited_order, unused_entries_count = reorder_bibliography_by_citation_order()
    
    if cited_order:
        print(f"\n=== 並べ替え完了 ===")
        print(f"引用順序:")
        for i, key in enumerate(cited_order, 1):
            print(f"  {i:2d}. {key}")
        print(f"\n引用された文献: {len(cited_order)} 件")
        print(f"未引用文献: {unused_entries_count} 件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reorder":
        reorder_only()
    else:
        main() 