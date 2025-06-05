# CRT-Nodes 

CRT-Nodes is a collection of custom nodes for ComfyUI. 

## Features

1. **Toggle Lora Unet Blocks L1**: 
   - This node allows you to toggle Lora Unet blocks at layer 1. 
   - It takes a string input and outputs a concatenated string of the active blocks.

2. **Toggle Lora Unet Blocks L2**: 
   - Similar to the L1 node but designed for layer 2. 
   - It also takes a string input and outputs a concatenated string of the active blocks.

3. **Remove Trailing Comma**: 
   - This node takes a string input and removes the last trailing comma, providing a clean string output for the Flux LoRA training "block_args".

4. **lora loader str**:
   - This is a simple LoRA Loader that can output the name of the model loaded as a string, with a switch to also embed the strength if wanted.

5. **Boolean transform node**:
   - Clamp list value to 1 if non zero

5. **Video duration calculator**:
   - Calculate video duration (in seconds) with fps and frame_count

5. **CRT Post Process Node**: https://www.youtube.com/watch?v=vZ8tgiSDf9Y
   - Complete AIO Post Process solution

![Capture d'écran 2025-06-06 012549](https://github.com/user-attachments/assets/35d84ff3-ba19-4cbe-89d3-30b366fdf78d)


7. **FluxTiledSamplerCustom**: Similar to UltimateSDUpscaler but with Advanced Inputs and better tile size handling 

8. **FancyNote**: A simple fancy Note with nice visual effects, a Text size slider and color picker to chose the theme

![Capture d'écran 2025-06-04 004532](https://github.com/user-attachments/assets/3eb1dbda-d092-463e-8f0e-97431b426080)

9. **Flux LoRA Blocks Patcher**: https://www.youtube.com/watch?v=uHIoEnLX38Q
   - Patch your Flux model whit LoRA applyed, works with multiple LoRA. Full Single/Double Blocks range, Presets, Randomizer.

![Capture d'écran 2025-06-06 012706](https://github.com/user-attachments/assets/84ee53c0-fda3-4983-8f2f-c851217b4bd9)

Example:

Toggle Lora Unet Blocks L1 Node
specify which blocks you want to toggle on.
The output will provide a concatenated string of the active blocks.

Toggle Lora Unet Blocks L2 Node
Similar to the L1 node, connect your input and specify blocks to toggle.

Remove Trailing Comma Node
Connect the input string that may have a trailing comma, and the output will be a cleaned string without the trailing comma.

lora loader str
You want to compair LoRA and combine the images output for comparison purpose (Ex: CR Simple Image Compare), you can use the string output of the lora loader as text input fore the compare node.

boolean transform node
Use it after "Mask Or Image To Weight" from KJNodes. It was made for logic purpose, and basically say (If a mask is detected, the boolean output will be "True", if not, "False". 


Contributing
Feel free to submit issues and pull requests. Contributions are welcome!
License
This project is licensed under the MIT License. See the LICENSE file for more details.
Acknowledgments
Thanks to the ComfyUI community for their support and contributions.

https://civitai.green/user/pgc
https://discord.gg/8wYS9MBQqp
