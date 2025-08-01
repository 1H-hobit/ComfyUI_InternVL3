// 导入应用实例和工具函数
import {app } from "../../scripts/app.js";

// 注册自定义扩展
app.registerExtension({
    name: "InternVL3_unload",  // 扩展名称

    // 在节点定义注册前的钩子函数
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 根据节点类型执行不同的自定义逻辑
        switch(nodeData.name){
            case 'InternVLModelLoader':  // 完整版文本翻译节点
			InternVL3_unload_widget(nodeType, nodeData, app);
                break;
        }
    },
});

// 修改 InternVL3.js 中的按钮回调逻辑
function InternVL3_unload_widget(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const onc = onNodeCreated?.apply(this, arguments);
        let node = this;

        node.addWidget("button", "卸载模型", "delete_model", function(value, widget, node){
            app.graph.setDirtyCanvas(true);

            // 修复：使用 app.callApi 调用后端自定义事件接口
            app.callApi(
                "custom_node_event",  // ComfyUI 自定义节点事件的默认接口
                {
                    node_id: node.id,    // 传递节点ID，后端用于识别节点
                    event: { action: "unload_model" }  // 事件数据
                },
                // 可选：处理后端响应
                (response) => {
                    console.log("模型卸载结果:", response);
                }
            );
        });
        return onc;
    };
}