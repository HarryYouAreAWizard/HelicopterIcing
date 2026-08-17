from pynput import mouse

def on_move(x, y):
    # print(f"Mouse moved to position: ({x}, {y})")
    pass

def on_click(x, y, button, pressed):
    action = "Pressed" if pressed else "Released"
    print(f"Button {button} {action} at ({x}, {y})")
    
    # Optional: Stop the listener by returning False when clicking right button
    if button == mouse.Button.right and not pressed:
        print("Stopping listener...")
        return False

def on_scroll(x, y, dx, dy):
    pass
    # direction = "Down/Right" if (dx < 0 or dy < 0) else "Up/Left"
    # print(f"Scrolled {direction} at ({x}, {y})")

# This starts a background thread to listen for mouse events
with mouse.Listener(
        on_move=on_move, 
        on_click=on_click, 
        on_scroll=on_scroll) as listener:
    listener.join()