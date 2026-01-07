### 【VScode】设置代理

[【VScode】设置代理，通过代理连接服务器](https://blog.csdn.net/qq_43633528/article/details/144903155)

打开`setting.json` 文件

![proxy](../图片/3ce4e66dcd4441779f9c3fdc27fc1f7e.png)

### 容器中使用copilot就是会有问题

[快速解决vscode远程连接时copilot提示脱机状态无法使用的问题](https://blog.csdn.net/messi10101010___/article/details/149963354)

只需要在设置(setting)中搜索"extension kind"，点击settings.json；

找到"remote.extensionKind"，加入如下"[Github](https://so.csdn.net/so/search?q=Github&spm=1001.2101.3001.7020)."开头的4行代码即可。

```
"remote.extensionKind": {
  "Github.copilot": ["ui"],
  "Github.copilot-chat": ["ui"],
  "Github.copilot-labs": ["ui"],
  "Github.copilot-chat-labs": ["ui"],
  "Github.copilot-chat-completions": ["ui"],
  "Github.copilot-chat-completions-labs": ["ui"],
  "Github.copilot-chat-completions-remote": ["ui"],
  "Github.copilot-chat-completions-remote-labs": ["ui"]
}
```

