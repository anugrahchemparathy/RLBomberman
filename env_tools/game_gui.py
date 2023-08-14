"""
Using tkinter, creates a gui that visualizes the game state and steps through it with
a new frame every refresh_time milliseconds.

This code was written heavily using GPT-3.5
"""
import tkinter as tk

class GameGUI:
    def __init__(self, game_states):
        self.game_states = game_states
        self.current_state_index = 0
        self.refresh_time = 100 # in milliseconds

        # Create the root window
        self.root = tk.Tk()

        # Create the label to display the current time step
        self.time_label = tk.Label(self.root, text="time step = 0")
        self.time_label.pack()

        # Create the canvas to display the game state
        self.canvas = tk.Canvas(self.root, width=510, height=510)
        self.canvas.pack()

        # Start the update loop and run the main loop
        self.update()
        self.root.mainloop()

    def update(self):
        if self.current_state_index >= len(self.game_states):
            # If the final game state has been displayed, stop scheduling further updates
            return

        # Get the current game state string
        state_str = self.game_states[self.current_state_index]

        # Convert the state string to a 2D list of emojis
        state_list = [[char for char in row] for row in state_str.split('\n')]

        # Clear the canvas
        self.canvas.delete('all')

        # Draw the state list on the canvas
        for i, row in enumerate(state_list):
            for j, emoji in enumerate(row):
                x = j * 30
                y = i * 30
                self.canvas.create_text(x+15, y+15, text=emoji, font=('Arial', 20))

        # Update the time step label
        self.time_label.config(text=f"time step = {self.current_state_index}")

        # Increment the current state index
        self.current_state_index += 1

        if self.current_state_index >= len(self.game_states):
            # If the final game state has been displayed, freeze the GUI
            self.root.mainloop()
        else:
            self.root.after(self.refresh_time, self.update)

if __name__ == "__main__":
    with open('replays/replay2.txt', 'r') as f:
        content = f.read()

    # Split the content of the file into a list of game state strings
    game_states= content.split('\n\n')

    gui = GameGUI(game_states)