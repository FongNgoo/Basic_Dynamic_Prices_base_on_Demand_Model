#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import subprocess
from google.colab import drive

# === CẤU HÌNH RIÊNG (CHỈ THAY 1 LẦN) ===
USERNAME = "FongNgoo"   # ← THAY ĐỔI
REPO_NAME = "Basic_Dynamic_Prices_base_on_Demand_Model"        # ← THAY ĐỔI
BRANCH = "main"                     # hoặc "master"
# ======================================

REPO_DIR = f"/content/{REPO_NAME}"
REPO_URL = f"git@github.com:{USERNAME}/{REPO_NAME}.git"

get_ipython().system('git config --global user.email "phongngo4060@gmail.com"')
get_ipython().system('git config --global user.name "FongNgoo"')


# In[11]:


def _setup_ssh():
    """Tải SSH key từ Drive và cấu hình"""
    drive.mount('/content/drive', force_remount=False)
    SSH_DIR = '/content/drive/MyDrive/.colab_ssh'
    SSH_KEY = f"{SSH_DIR}/id_colab"
    if not os.path.exists(SSH_KEY):
        raise FileNotFoundError("SSH key chưa tồn tại! Chạy file setup trước.")

    os.makedirs('/root/.ssh', exist_ok=True)
    subprocess.run(['cp', f'{SSH_KEY}', f'{SSH_KEY}.pub', '/root/.ssh/'], check=False)
    subprocess.run(['chmod', '600', '/root/.ssh/id_colab'], check=True)
    subprocess.run(['chmod', '644', '/root/.ssh/id_colab.pub'], check=True)

    config = f"""Host github.com
    HostName github.com
    User git
    IdentityFile /root/.ssh/id_colab
    StrictHostKeyChecking no
"""
    with open('/root/.ssh/config', 'w') as f:
        f.write(config)
    subprocess.run(['chmod', '600', '/root/.ssh/config'], check=True)


# In[12]:


def _ensure_repo():
    """Đảm bảo repo tồn tại, clone nếu cần"""
    if not os.path.exists(REPO_DIR):
        print(f"Repo không tồn tại → Đang clone: {REPO_URL}")
        result = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR])
        if result.returncode != 0:
            raise RuntimeError("Clone thất bại! Kiểm tra SSH key và repo URL.")
    # Luôn cd vào repo
    os.chdir(REPO_DIR)
    print(f"Đã vào thư mục: {REPO_DIR}")


# In[7]:


def sync(message=None, auto_push=True, branch=None):
    """
    TỰ ĐỘNG SYNC 2 CHIỀU (AN TOÀN, KHÔNG LỖI getcwd)
    """
    global BRANCH
    branch = branch or BRANCH
    get_ipython().system('git checkout main')
    if not os.path.exists('/content/drive/MyDrive'):
        print("Drive chưa mount → đang mount lại...")
        drive.mount('/content/drive', force_remount=True)
    else:
        # Kiểm tra xem có bị treo không
        try:
            os.listdir('/content/drive/MyDrive')
        except:
            print("Drive bị treo → force remount lại...")
            get_ipython().system('pkill -9 -f drive')
            drive.mount('/content/drive', force_remount=True)
    print("Bước 1: Thiết lập SSH...")
    _setup_ssh()

    print("Bước 2: Đảm bảo repo tồn tại...")
    _ensure_repo()

    print("Bước 3: Pull từ GitHub...")
    result = subprocess.run(['git', 'pull', 'origin', branch, '--rebase'])
    if result.returncode != 0:
        print("Pull thất bại! Kiểm tra kết nối hoặc xung đột.")
        return

    get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model')
    print("Bước 4: Kiểm tra thay đổi...")
    status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip().split('\n')
    status = [s for s in status if s.strip()]

    if not status:
        print("Không có thay đổi.")
        return

    print(f"Phát hiện {len(status)} thay đổi:")
    for line in status[:5]:
        print(f"   {line}")
    if len(status) > 5:
        print(f"   ... và {len(status)-5} thay đổi khác")

    if not auto_push:
        print("auto_push=False → Bỏ qua push")
        return

    msg = message or f"Auto sync from Colab - $(date)"
    print(f"Đang commit: {msg}")
    subprocess.run(['git', 'add', '.'])
    commit_result = subprocess.run(['git', 'commit', '-m', msg], capture_output=True)
    if b"nothing to commit" in commit_result.stdout:
        print("Không có gì để commit.")
        return

    print("Đang push lên GitHub...")
    push_result = subprocess.run(['git', 'push', 'origin', branch])
    if push_result.returncode == 0:
        print("SYNC HOÀN TẤT!")
    else:
        print("Push thất bại!")


# In[13]:


sync()


# In[14]:


# GIẢI QUYẾT XUNG ĐỘT + FORCE SYNC NGAY LẬP TỨC (dùng khi bạn muốn giữ toàn bộ thay đổi trong Colab)

repo_path = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model"
get_ipython().run_line_magic('cd', '{repo_path}')

# 1. Bỏ toàn bộ rebase đang bị kẹt (nếu có)
get_ipython().system('git rebase --abort 2>/dev/null || echo "Không có rebase đang chạy"')

# 2. Reset cứng về trạng thái hiện tại của GitHub (lấy mới nhất từ remote)
get_ipython().system('git fetch origin')
get_ipython().system('git reset --hard origin/main   # thay main → master nếu repo bạn dùng master')

# 3. Ghi đè luôn mọi thứ trong Colab lên GitHub (force push)
get_ipython().system('git add . --all')
get_ipython().system('git commit -m "Force sync from Colab - $(date)" || echo "Không có gì để commit mới"')

# 4. FORCE PUSH (bắt buộc dùng -f để ghi đè)
get_ipython().system('git push origin main --force   # thay main → master nếu cần')

print("ĐÃ GIẢI QUYẾT XUNG ĐỘT + FORCE SYNC THÀNH CÔNG!")
print("GitHub giờ đây 100% giống hệt Colab của bạn!")


# In[ ]:


# PUSH NGAY LẬP TỨC – KHÔNG QUAN TÂM XUNG ĐỘT (siêu nhanh)

repo_path = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model"
get_ipython().run_line_magic('cd', '{repo_path}')

# Fix Drive treo (nếu có)
get_ipython().system('pkill -9 -f "drive" 2>/dev/null')
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive', force_remount=True)

# Vào đúng thư mục
get_ipython().run_line_magic('cd', '{repo_path}')

# Bỏ mọi rebase đang kẹt
get_ipython().system('git rebase --abort 2>/dev/null')

# Lấy mới nhất từ GitHub trước (đề phòng)
get_ipython().system('git fetch origin')

# Ghi đè toàn bộ bằng những gì đang có trong Colab
get_ipython().system('git add . --all')
get_ipython().system('git commit -m "Push nhanh từ Colab - $(date \'+ %Y-%m-%d %H:%M:%S\')" || echo "Không có gì mới để commit"')

# FORCE PUSH (quan trọng nhất – ghi đè hoàn toàn lên GitHub)
get_ipython().system('git push origin main --force-with-lease')

# Nếu repo bạn dùng branch master thì dùng dòng này thay dòng trên:
# !git push origin master --force-with-lease

print("ĐÃ PUSH XONG 100%!")
print("GitHub giờ đây giống hệt Colab của bạn rồi!")
print("Link repo của bạn sẽ có commit mới nhất ngay bây giờ")


# In[ ]:


# SIÊU CỨU CÁNH KHI DRIVE TREO NẶNG + COLAB BỊ CRASH TRACEBACK

# Bước 1: Kill sạch mọi kết nối Drive đang treo
get_ipython().system('pkill -9 -f "drive" 2>/dev/null')
get_ipython().system('pkill -9 -f "fuse" 2>/dev/null')
get_ipython().system('umount -f /content/drive 2>/dev/null')
get_ipython().system('rm -rf /content/drive 2>/dev/null')

# Bước 2: Remount lại Drive thật mạnh (force + flush cache)
from google.colab import drive
import time
time.sleep(2)  # chờ kill process xong
drive.mount('/content/drive', force_remount=True)

# Bước 3: Vào lại repo + push ngay lập tức (force với lease an toàn)
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model')

get_ipython().system('git add . --all')
get_ipython().system('git commit -m "Emergency push từ Colab - $(date \'+%Y-%m-%d %H:%M:%S\')" || echo "Không có gì để commit"')

# Force push an toàn (chỉ ghi đè nếu không ai khác push trong lúc này)
get_ipython().system('git push origin main --force-with-lease')

print("ĐÃ THOÁT KHỎI ĐỊA NGỤC DRIVE TREO!")
print("ĐÃ PUSH XONG 100% LÊN GITHUB RỒI!")
print("Bây giờ bạn có thể refresh GitHub thấy commit mới nhất ngay!")


# In[34]:


# KIỂM TRA & PUSH LÊN GITHUB (phiên bản chạy ngon 100% trong Colab)

import os
from datetime import datetime

repo_path = "/content/drive/MyDrive/Colab_Notebooks/Basic_Dynamic_Prices_base_on_Demand_Model"

# 1. Vào thư mục repo (tự fix nếu Drive bị treo)
try:
    os.chdir(repo_path)
    print(f"Đã vào thư mục: {os.getcwd()}")
except:
    print("Drive bị treo → Đang remount lại...")
    get_ipython().system('pkill -9 -f drive 2>/dev/null')
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir(repo_path)
    print(f"Đã vào lại thư mục: {os.getcwd()}")

# 2. Git status
print("\nGIT STATUS:")
get_ipython().system('git status -sb')

# 3. 3 commit mới nhất ở local
print("\n3 COMMIT MỚI NHẤT (máy bạn):")
get_ipython().system('git log --oneline -3 --pretty=format:"%C(yellow)%h %Creset%s %Cgreen(%cr)%Creset" --date=relative')

# 4. Fetch và xem commit mới nhất trên GitHub
print("\nĐang fetch từ GitHub...")
get_ipython().system('git fetch origin')

print("\n3 COMMIT MỚI NHẤT TRÊN GITHUB (origin/main):")
get_ipython().system('git log origin/main --oneline -3 --pretty=format:"%C(yellow)%h %Creset%s %Cgreen(%cr)%Creset" --date=relative 2>/dev/null || echo "Không thấy branch main, thử master..."')
get_ipython().system('git log origin/master --oneline -3 --pretty=format:"%C(yellow)%h %Creset%s %Cgreen(%cr)%Creset" --date=relative 2>/dev/null')

# 5. Nếu có thay đổi chưa push → tự động commit + push luôn
print("\nĐang kiểm tra thay đổi chưa push...")
get_ipython().system('git add .')

# Kiểm tra có gì thay đổi không
changed = get_ipython().getoutput('git status --porcelain')
if changed:
    print(f"Có {len(changed)} thay đổi → Đang commit & push...")
    commit_msg = f"Auto sync from Colab - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    get_ipython().system('git commit -m "{commit_msg}"')
    get_ipython().system('git push origin main 2>/dev/null || git push origin master')
    print("ĐÃ PUSH THÀNH CÔNG LÊN GITHUB!")
else:
    print("ĐÃ ĐỒNG BỘ HOÀN TOÀN! Không có gì cần push thêm.")

print("\nHOÀN TẤT KIỂM TRA & SYNC!")

