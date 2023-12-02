import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QCheckBox, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class NoiseGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.left_channel_enabled = True
        self.right_channel_enabled = True
        self.is_paused = False
        self.initUI()

    def initUI(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Matplotlib Figures for each channel
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Open File Button
        self.open_file_button = QPushButton('Open WAV File', self)
        self.open_file_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.open_file_button)

        # Play Button
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.start_stream)
        layout.addWidget(self.play_button)

        # Pause Button
        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_stream)
        layout.addWidget(self.pause_button)

        # Continue Button
        self.continue_button = QPushButton('Continue', self)
        self.continue_button.clicked.connect(self.continue_stream)
        layout.addWidget(self.continue_button)

        # Stop Button
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_stream)
        layout.addWidget(self.stop_button)

        # Channel Checkboxes
        self.left_channel_checkbox = QCheckBox('Enable Left Channel (Noise)')
        self.left_channel_checkbox.setChecked(True)
        self.left_channel_checkbox.stateChanged.connect(self.toggle_channels)
        layout.addWidget(self.left_channel_checkbox)

        self.right_channel_checkbox = QCheckBox(
            'Enable Right Channel (Anti-Noise)')
        self.right_channel_checkbox.setChecked(True)
        self.right_channel_checkbox.stateChanged.connect(self.toggle_channels)
        layout.addWidget(self.right_channel_checkbox)

        # Initialize plots
        self.ax1.set_title('Noise Signal (Left Channel)')
        self.ax2.set_title('Anti-phase Signal (Right Channel)')
        self.ax3.set_title('Mixed Signal')
        self.plot_lines = [self.ax1.plot(np.zeros(1000))[0],
                           self.ax2.plot(np.zeros(1000))[0],
                           self.ax3.plot(np.zeros(1000))[0]]

        # Timer for updating plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer_interval = 100  # milliseconds

        self.setLayout(layout)
        self.setWindowTitle('Real-Time Audio Signals')

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open WAV File", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.song_file = file_name

    def start_stream(self):
        if not self.song_file:
            print("Please select a WAV file first.")
            return

        self.song, self.song_rate = sf.read(self.song_file, dtype='float32')
        if self.song.ndim == 1:  # Mono file
            self.song = np.stack((self.song, self.song), axis=-1)

        self.current_frame = 0
        self.frame_size = 1024  # Number of frames per chunk

        # Start streaming
        self.stream = sd.OutputStream(
            samplerate=self.song_rate, channels=2, callback=self.audio_callback)
        self.stream.start()
        self.timer.start(self.timer_interval)
        self.is_paused = False

    def pause_stream(self):
        self.is_paused = True

    def continue_stream(self):
        self.is_paused = False

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.timer.stop()

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.is_paused:
            outdata.fill(0)
            return

        chunk_start = self.current_frame
        chunk_end = chunk_start + frames
        chunk = self.song[chunk_start:chunk_end]

        if len(chunk) < frames:
            looped_chunk = np.zeros((frames, 2), dtype=self.song.dtype)
            looped_chunk[:len(chunk)] = chunk
            chunk = looped_chunk

        # Apply channel enable/disable
        if not self.left_channel_enabled and not self.right_channel_enabled:
            chunk.fill(0)
        elif not self.left_channel_enabled:
            chunk[:, 0] = 0
            chunk[:, 1] = -chunk[:, 1]  # Inverting right channel
        elif not self.right_channel_enabled:
            chunk[:, 0] = -chunk[:, 0]  # Inverting left channel
            chunk[:, 1] = 0
        else:
            # If both channels are enabled, the right channel is the inverse of the left
            chunk[:, 1] = -chunk[:, 0]

        outdata[:] = chunk
        self.current_frame = chunk_end % len(self.song)

    def toggle_channels(self):
        self.left_channel_enabled = self.left_channel_checkbox.isChecked()
        self.right_channel_enabled = self.right_channel_checkbox.isChecked()

    def update_plots(self):
        if self.is_paused or not self.stream:
            return

        start = max(0, self.current_frame - 1000)
        end = start + 1000
        segment = self.song[start:end]

        if len(segment) == 0:
            return

        if segment.shape[0] < 1000:
            segment = np.pad(
                segment, ((0, 1000 - segment.shape[0]), (0, 0)), mode='constant')

        # Update plots based on the enabled/disabled state of the channels
        left_channel_data = segment[:, 0] if self.left_channel_enabled else np.zeros(
            1000)
        right_channel_data = - \
            segment[:, 0] if self.right_channel_enabled else np.zeros(1000)

        self.plot_lines[0].set_ydata(left_channel_data)
        self.plot_lines[1].set_ydata(right_channel_data)
        self.plot_lines[2].set_ydata(
            left_channel_data + right_channel_data)  # Mixed signal

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    ex = NoiseGenerator()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
