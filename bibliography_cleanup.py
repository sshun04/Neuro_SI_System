#!/usr/bin/env python3
"""
文献整理スクリプト：使用されていない文献をアーカイブに移動し、引用エラーを修正
"""

import re
import os
from pathlib import Path
from datetime import datetime

# 使用されていない文献のリスト（citation_analysis.pyの結果から）
UNUSED_CITATIONS = [
    'He2021fNIRS', 'agostinelli2024simul', 'dangelo2013realistic', 'deco2015rethinking',
    'doi2024evaluation', 'elbayad2020efficient', 'friederici2011brain', 'gile1997conference',
    'gonzalez2016higher', 'gonzalez2016role', 'guo2024sillm', 'he2016interpretese',
    'ishizuka2024two', 'jiao2022partially', 'kahneman1973attention', 'kuperberg2016comprehending',
    'lambon2010angular', 'li2022petit', 'lin2018costly', 'ma2024nast',
    'mathur2020tangled', 'mcdougal2016reproducibility', 'moser2013simultaneous', 'papi2023attention',
    'plevoets2018cognitive', 'pliatsikas2020language', 'seamless2023m4t', 'seamless2023seamless',
    'seghier2013angular', 'sulpizio2020bilingual', 'vaswani2017attention', 'wickens1984processing',
    'wickens2002multiple', 'zhang2024streamspeech'
]

def extract_bib_entry(content, entry_key):
    """指定されたキーのbibエントリを抽出"""
    pattern = rf'(@[a-zA-Z]+\{{{re.escape(entry_key)},.*?\n\}})'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1) if match else None

def remove_bib_entry(content, entry_key):
    """指定されたキーのbibエントリを削除"""
    pattern = rf'@[a-zA-Z]+\{{{re.escape(entry_key)},.*?\n\}}\n*'
    return re.sub(pattern, '', content, flags=re.DOTALL)

def backup_file(file_path):
    """ファイルのバックアップを作成"""
    backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(file_path):
        os.system(f'cp "{file_path}" "{backup_path}"')
        print(f"バックアップ作成: {backup_path}")

def move_unused_citations():
    """使用されていない文献をcites.bibからcites_archive.bibに移動"""
    
    cites_bib_path = "bib/cites.bib"
    archive_bib_path = "bib/cites_archive.bib"
    
    # バックアップ作成
    backup_file(cites_bib_path)
    backup_file(archive_bib_path)
    
    # cites.bibの内容を読み込み
    try:
        with open(cites_bib_path, 'r', encoding='utf-8') as f:
            cites_content = f.read()
    except FileNotFoundError:
        print(f"Error: {cites_bib_path} not found")
        return False
    
    # cites_archive.bibの内容を読み込み（存在しない場合は空文字列）
    try:
        with open(archive_bib_path, 'r', encoding='utf-8') as f:
            archive_content = f.read()
    except FileNotFoundError:
        archive_content = "% Archive of unused bibliography entries\n% These entries exist in the bibliography but are not cited in the main text\n\n"
    
    moved_entries = []
    
    # 使用されていない文献を移動
    for entry_key in UNUSED_CITATIONS:
        entry = extract_bib_entry(cites_content, entry_key)
        if entry:
            # アーカイブに追加
            archive_content += f"{entry}\n\n"
            # メインファイルから削除
            cites_content = remove_bib_entry(cites_content, entry_key)
            moved_entries.append(entry_key)
            print(f"移動: {entry_key}")
        else:
            print(f"Warning: {entry_key} not found in cites.bib")
    
    # ファイルを更新
    try:
        with open(cites_bib_path, 'w', encoding='utf-8') as f:
            f.write(cites_content)
        
        with open(archive_bib_path, 'w', encoding='utf-8') as f:
            f.write(archive_content)
        
        print(f"\n成功: {len(moved_entries)} 件の文献をアーカイブに移動しました")
        return True
        
    except Exception as e:
        print(f"Error writing files: {e}")
        return False

def fix_seamless_citation():
    """seamless2023v2をbarrault2023seamlessm4tに修正"""
    
    tex_file = "src/4_machine_translation_models.tex"
    
    # バックアップ作成
    backup_file(tex_file)
    
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # seamless2023v2をbarrault2023seamlessm4tに置換
        if 'seamless2023v2' in content:
            updated_content = content.replace('seamless2023v2', 'barrault2023seamlessm4t')
            
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"修正: seamless2023v2 → barrault2023seamlessm4t in {tex_file}")
            return True
        else:
            print("seamless2023v2 not found in the file")
            return False
            
    except Exception as e:
        print(f"Error fixing citation: {e}")
        return False

def generate_cleanup_report():
    """整理結果のレポートを生成"""
    
    report = f"""# 文献整理レポート

## 実行日時
{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 実行内容

### 1. 使用されていない文献のアーカイブ移動
- **移動対象**: {len(UNUSED_CITATIONS)} 件
- **移動先**: `bib/cites_archive.bib`
- **移動元**: `bib/cites.bib`

### 移動された文献一覧
"""
    
    for i, citation in enumerate(UNUSED_CITATIONS, 1):
        report += f"{i:2d}. `{citation}`\n"
    
    report += """
### 2. 引用エラーの修正
- `seamless2023v2` → `barrault2023seamlessm4t`
- ファイル: `src/4_machine_translation_models.tex`

## 結果
- `cites.bib` の文献数が減少し、管理が容易になりました
- 引用エラーが解決され、LaTeXコンパイルが正常に動作するはずです
- 不要な文献は `cites_archive.bib` に保存されており、必要時に復元可能です

## 注意事項
- 変更前のファイルは `.backup.YYYYMMDD_HHMMSS` 形式でバックアップされています
- 問題が発生した場合は、バックアップファイルから復元してください
"""
    
    with open("bibliography_cleanup_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("整理レポートを bibliography_cleanup_report.md に保存しました")

def main():
    print("文献整理を開始します...\n")
    
    # 1. 使用されていない文献をアーカイブに移動
    print("=== 使用されていない文献をアーカイブに移動 ===")
    move_success = move_unused_citations()
    
    print("\n=== 引用エラーの修正 ===")
    # 2. seamless2023v2の問題を修正
    fix_success = fix_seamless_citation()
    
    # 3. レポート生成
    print("\n=== レポート生成 ===")
    generate_cleanup_report()
    
    if move_success and fix_success:
        print("\n✅ 文献整理が正常に完了しました")
        print("LaTeXの再コンパイルを行って、エラーがないことを確認してください")
    else:
        print("\n⚠️ 一部の処理でエラーが発生しました。バックアップファイルを確認してください")

if __name__ == "__main__":
    main() 