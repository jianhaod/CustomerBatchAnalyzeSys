Customer Batch Analyze System
System use camera stream video to analyze and detect how many batch custom come into monitor
feature list:
1. fetch video and split frame for dectect model
2. use dectect model to analyze image, get customer feature
3. integrate log module, enable log module test case
4. enable yaml module to support load sys config, enable yaml test case
5. add system lanuch entry script
6. add fetch file list support, enable fetch local video list test
7. multiple process to fetch video and push dic into queue, enable multiple consumer process get dic from queue
8. caffe net person detect feature support & enable detect func testing
9. add thirdparty lib such as caffe-fast-rcnn support sys
