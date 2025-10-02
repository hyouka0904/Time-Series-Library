import os
import subprocess

# GitHub 限制
WARNING_LIMIT = 50 * 1024 * 1024   # 50 MB
ERROR_LIMIT = 100 * 1024 * 1024    # 100 MB

# 取得 Git 歷史中所有 blob 物件 (檔案)
result = subprocess.run(
    ["git", "rev-list", "--objects", "--all"],
    capture_output=True, text=True, check=True
)
objects = result.stdout.splitlines()

# 查詢 blob 大小
result = subprocess.run(
    ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
    input="\n".join(o.split()[0] for o in objects),
    capture_output=True, text=True, check=True
)

sizes = {}
for line in result.stdout.splitlines():
    oid, otype, size = line.split()
    if otype == "blob":
        sizes[oid] = int(size)

# 建立檔案 -> 大小清單
file_sizes = []
for line in objects:
    parts = line.split()
    if len(parts) == 2:  # 格式: <blob-id> <檔案路徑>
        oid, path = parts
        if oid in sizes:
            file_sizes.append((path, sizes[oid]))

# 排序 (由大到小)
file_sizes.sort(key=lambda x: x[1], reverse=True)

# 顯示前 10 大
print("📦 Git 歷史中前 10 大檔案：")
for i, (path, size) in enumerate(file_sizes[:10], start=1):
    status = ""
    if size > ERROR_LIMIT:
        status = "❌ 超過 100MB，無法推送"
    elif size > WARNING_LIMIT:
        status = "⚠️ 超過 50MB，推送會警告"
    print(f"{i}. {path} - {size/1024/1024:.2f} MB {status}")

# 總結
too_large = any(size > ERROR_LIMIT for _, size in file_sizes)
if not too_large:
    print("\n✅ 歷史紀錄中沒有超過 100MB 的檔案，可以推送到 GitHub")
else:
    print("\n🚨 歷史紀錄中存在超過 100MB 的檔案，必須用 git filter-repo 或 git filter-branch 移除")
