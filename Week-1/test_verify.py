
import os
import streamlit.components.v1 as components
script_dir = r'g:\Projects\Skill Nexis\DL-AI\skill-nexis-dl-ai\Week-1'
comp_dir = os.path.join(script_dir, 'canvas_component')
comp = components.declare_component('digit_canvas', path=comp_dir)
print('Component verified in file context:', comp)
