# Prompt 檔案命名規範

本文件定義了專案中 prompt 檔案的命名規則和版本控制方式。

## 基本命名格式

Prompt 檔案應遵循以下命名格式：

```
prompt{LLM編號}_v{大版本}.{小版本}.md
```

例如：
- `prompt2_v1.0.md` 表示 LLM #2 的 prompt 初始版本
- `prompt1_v2.1.md` 表示 LLM #1 的 prompt 第二大版本的第一次小修改

## 版本號規則

版本號由兩部分組成：大版本和小版本。

### 大版本 (Major Version)
- 當 prompt 有重大變動時才更新大版本號
- 重大變動包括：邏輯結構改變、目標功能變更、完全重寫等

### 小版本 (Minor Version)
- 當進行小規模調整、優化或重構時更新小版本號
- 例如：文字潤飾、格式調整、改善提示詞精確度等

## 範例

- `prompt1_v1.0.md` - LLM #1 的初始 prompt
- `prompt1_v1.1.md` - 在初始版本上進行小修改
- `prompt1_v2.0.md` - LLM #1 的重大修改版本