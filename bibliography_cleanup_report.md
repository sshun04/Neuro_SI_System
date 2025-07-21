# 文献整理レポート

## 実行日時
2025年07月21日 22:03:28

## 実行内容

### 1. 使用されていない文献のアーカイブ移動
- **移動対象**: 34 件
- **移動先**: `bib/cites_archive.bib`
- **移動元**: `bib/cites.bib`

### 移動された文献一覧
 1. `He2021fNIRS`
 2. `agostinelli2024simul`
 3. `dangelo2013realistic`
 4. `deco2015rethinking`
 5. `doi2024evaluation`
 6. `elbayad2020efficient`
 7. `friederici2011brain`
 8. `gile1997conference`
 9. `gonzalez2016higher`
10. `gonzalez2016role`
11. `guo2024sillm`
12. `he2016interpretese`
13. `ishizuka2024two`
14. `jiao2022partially`
15. `kahneman1973attention`
16. `kuperberg2016comprehending`
17. `lambon2010angular`
18. `li2022petit`
19. `lin2018costly`
20. `ma2024nast`
21. `mathur2020tangled`
22. `mcdougal2016reproducibility`
23. `moser2013simultaneous`
24. `papi2023attention`
25. `plevoets2018cognitive`
26. `pliatsikas2020language`
27. `seamless2023m4t`
28. `seamless2023seamless`
29. `seghier2013angular`
30. `sulpizio2020bilingual`
31. `vaswani2017attention`
32. `wickens1984processing`
33. `wickens2002multiple`
34. `zhang2024streamspeech`

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
