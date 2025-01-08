## install gh
```
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
gh auth login
```

## follow git check-in commands
```
git init
git config --global --add safe.directory git config --global --add safe.directory /mnt/batch/tasks/shared/LS_root/mounts/clusters/cpuds11001/code/Users/pupanda/gpt-use-case/foundation-models/quantized-model-inference
git add .
git status
git config --global user.email "pupanda@outlook.com"
git config --global user.name "Purna Chandra Panda"
#rm -rf .git # if removing earlier git init
git commit -m "initial commit"
gh repo create quantized-model-inference --public --source=. --remote=origin
git branch -M main
git push -u origin main
```
## recurring check-in
```
git status
git add .
git status
git commit -m "xxx"
git push -u origin main
```
