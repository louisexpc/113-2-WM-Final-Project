#!/bin/bash

# 使用方式：
# ./gdrive_batch_mixed_download.sh file_list.txt --output-dir ./downloads

# 參數解析
LIST_FILE="$1"
shift
OUTPUT_DIR="."
SIZE_LIMIT_MB=100  # 超過這個值的會用 gdown

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "❌ 未知參數: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# 檢查 gdown 是否存在
if ! command -v gdown >/dev/null 2>&1; then
  echo "⚠️  尚未安裝 gdown，將無法下載大檔案 (> ${SIZE_LIMIT_MB}MB)"
  echo "可使用：pip install gdown"
fi

# 暫存檔
COOKIE_FILE=$(mktemp)
HTML_FILE=$(mktemp)

download_with_curl() {
  FILE_ID=$1
  FILE_NAME=$2
  DEST_PATH="${OUTPUT_DIR}/${FILE_NAME}"

  echo "📥 (curl) Downloading ${FILE_NAME}..."

  curl -c $COOKIE_FILE -s -L \
    "https://drive.google.com/uc?export=download&id=${FILE_ID}" > $HTML_FILE

  CONFIRM=$(awk '/download/ {print $NF}' $HTML_FILE | sed 's/.*confirm=\(.*\)&id=.*/\1/')

  curl -Lb $COOKIE_FILE \
    "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
    -o "$DEST_PATH"

  echo "✔ Saved to: $DEST_PATH"
}

download_with_gdown() {
  FILE_ID=$1
  FILE_NAME=$2
  DEST_PATH="${OUTPUT_DIR}/${FILE_NAME}"

  echo "📥 (gdown) Downloading ${FILE_NAME}..."
  gdown --id "$FILE_ID" --output "$DEST_PATH"
  echo "✔ Saved to: $DEST_PATH"
}

# 主程式：依序讀取每一行
while read -r FILE_ID FILE_NAME; do
  [[ -z "$FILE_ID" || -z "$FILE_NAME" ]] && continue

  # 嘗試用 curl 抓 headers 檢查檔案大小（以 gdown API 會更精準，但我們用 curl 快速估算）
  SIZE_BYTES=$(curl -sI "https://drive.google.com/uc?export=download&id=${FILE_ID}" | grep -i Content-Length | awk '{print $2}' | tr -d '\r')

  if [[ "$SIZE_BYTES" =~ ^[0-9]+$ ]]; then
    SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
    if (( SIZE_MB > SIZE_LIMIT_MB )); then
      download_with_gdown "$FILE_ID" "$FILE_NAME"
    else
      download_with_curl "$FILE_ID" "$FILE_NAME"
    fi
  else
    echo "⚠️  無法確認大小，預設用 curl 嘗試：$FILE_NAME"
    download_with_curl "$FILE_ID" "$FILE_NAME"
  fi

done < "$LIST_FILE"

# 清理暫存
rm -f $COOKIE_FILE $HTML_FILE
