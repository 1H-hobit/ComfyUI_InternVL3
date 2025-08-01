// 导入应用实例和工具函数
import { app } from "../../scripts/app.js";

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

            // === 使用正确的端点和方法 ===
            // 获取 CSRF 令牌（如果可用）
            const csrfToken = app.csrf_token || 
                (document.querySelector('meta[name="csrf-token"]')?.content || "");
            
            // 使用 ComfyUI 的标准节点事件端点
            fetch("/object_info/" + node.type, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "X-CSRFToken": csrfToken
                },
                body: JSON.stringify({
                    action: "unload_model",  // 自定义动作
                    node_id: node.id         // 节点ID
                })
            })
            .then(response => {
                if (response.ok) {
                    console.log("✅ 模型卸载成功");
                    return response.json();
                } else {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
            })
            .then(data => {
                if (data && data.error) {
                    console.error("❌ 模型卸载失败:", data.error);
                } else {
                    console.log("✅ 模型已卸载:", data);
                }
            })
            .catch(error => {
                console.error("❌ API请求错误:", error);
            });
        });
        return onc;
    };
}