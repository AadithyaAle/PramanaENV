import gradio as gr
import subprocess

def run_agent():
    try:
        # Runs your inference script and captures the terminal output
        result = subprocess.run(["python3", "inference.py"], capture_output=True, text=True, timeout=60)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return f"Error running agent: {str(e)}"

# Build a beautiful UI for the judges
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧹 Meta OpenEnv: Self-Healing Data Cleaner")
    gr.Markdown("Click the button below to unleash the Qwen 72B Agent on our stringent, ML-validated data environment. Watch it navigate dirty data, apply fixes, and submit for PyTorch validation!")
    
    with gr.Row():
        start_btn = gr.Button("🚀 Run AI Agent", variant="primary")
    
    output_log = gr.Code(label="Agent Terminal Output", language="json", lines=20)
    
    start_btn.click(fn=run_agent, outputs=output_log)

demo.launch()