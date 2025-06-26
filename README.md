# 使用说明
本实验基于sagemaker notebook g5.xlarge gpu 机器开发。如果使用vscode server cpu运行该实验时有需要调整的地方。


# 实验步骤
1. 进入到Cloudformation 控制台，上传sagemaker-bedrock-notebook.yaml 文件，创建一个cloudformation stack。该stack会创建一个Sagemaker notebook 笔记本，以及配置具有admin权限的role.

2. 进入Sagemaker控制台，打开notebook, 下载实验代码：

git clone https://github.com/isoyaoya/Component-level-agent-evaluation-with-deepeval.git

3. 运行文件路径下的笔记本来完成实验
