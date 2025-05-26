# 113-2 深度學習之生醫影像分析實務 期末報告

## 環境安裝

### POETRY

先安裝poetry
[text](https://blog.kyomind.tw/python-poetry/)

接著執行以下命令來建立虛擬環境，以及安裝所需套件

```shell
poetry install
```

### 選擇 VS Code Workspace Interpreter

為了在 VS Code 中使用該虛擬環境，請依照以下步驟設定：

- 按下快捷鍵 Ctrl+Shift+P 呼叫命令面板。
- 輸入並選擇「Python: Select Interpreter」。
- 從清單中選擇 Poetry 虛擬環境所對應的 Python interpreter。如果未顯示，可能需重新開啟 VS Code 或手動設定 interpreter path。

## 分支

如果你想增加新檔案，又害怕把別人想的東西搞砸，可以使用分支功能

### 檢查現有分支

在終端機中輸入以下命令，可查看目前所有分支及目前所在的分支：

```shell
git branch
```

### 建立新分支

要建立一個新的分支，可以使用：

```shell
git branch new-branch-name
```

這會基於目前所在的分支建立一個新分支。

### 切換至新分支

建立新分支後，可以使用以下命令切換到該分支：

```shell
git checkout new-branch-name
```

或者可以直接建立並切換至新分支：

```shell
git checkout -b new-branch-name
```

### 提交修改

在新分支中做完修改後，先加入到暫存區再進行提交：

```shell
git add .
git commit -m "你的修改說明"
```

### 合併分支

當您準備好將新分支的修改合併回主分支（例如 master 或 main），請先切換回目標分支：

```shell
git checkout main
```

接著合併新分支：

```shell
git merge new-branch-name
```

### 刪除分支

合併完成後，若不再需要該分支，可刪除：

```shell
git branch -d new-branch-name
```

### 解決衝突

如合併時遇到衝突，Git 會提示您手動修改檔案。請根據提示編輯檔案，解決衝突後執行：

```shell
git add 衝突解決的檔案
git commit
```
