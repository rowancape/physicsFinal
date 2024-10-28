import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt

# Initial streamlit configuration
st.set_page_config(page_title="Test", layout="wide")

# Reading data into a pandas DataFrame from csv
df = pd.read_csv("MainLinearAcceleration.csv")
gps = pd.read_csv("MainLocation.csv")

# Remove first and last 25 seconds to remove messy data
max_time = df["Time (s)"].max()
df = df[(df["Time (s)"] > 25) & (df["Time (s)"] < max_time - 25)]
df.reset_index(drop=True, inplace=True)

max_time_gps = gps["Time (s)"].max()
gps = gps[(gps["Time (s)"] > 25) & (gps["Time (s)"] < max_time_gps - 25)]
gps.reset_index(drop=True, inplace=True)

# Lowpass filter
def butterLowPassFilt(data, cutoff, fs, nyq, order):
  normalCutoff = cutoff/nyq
  b, a = butter(order, normalCutoff, btype = "low", analog=False)
  y = filtfilt(b, a, data)
  return y

# Producing frequency spectrum data using fast fourier transform
time = df["Time (s)"].values
signal = df["Linear Acceleration z (m/s^2)"].values
N = len(signal)
T = time[1] - time[0]
fs = 1 / T
magnitudes = (np.abs(np.fft.fft(signal)) / N)[1:(N // 2)]
frequencies = np.fft.fftfreq(N, T)[1:(N // 2)]
freqSpec = pd.DataFrame(list(zip(frequencies, magnitudes)), columns=['Frequencies', 'Magnitudes'])

# Producing Folium map
points = [
  [gps["Latitude (°)"][i], gps["Longitude (°)"][i]] for i in range(len(gps))
]
m = folium.Map(location=[gps["Latitude (°)"].mean(), gps["Longitude (°)"].mean()], zoom_start=15)
folium.PolyLine(locations=points, smooth_factor=0).add_to(m)

# Rendering in Streamlit
st.markdown("# Estimating step count from linear accelerometer data using two different methods")

st.markdown("## 1. Estimating step count by lowpass filtration")

st.markdown("### Provided sliders if you want to play with filter values")
T = df["Time (s)"][len(df["Time (s)"]) - 1] - df["Time (s)"][0]
n = len(df["Time (s)"])
fs = n/T
nyq = st.slider("Nyquist frequency", fs/0.5, fs/5, fs/2)
order = st.slider("Order value", 1, 10, 3)
cutoff = st.slider("Cutoff value", 1/(0.1), 1/(0.9), 1/(0.4))
df["filtered_z"] = butterLowPassFilt(df["Linear Acceleration z (m/s^2)"], cutoff, fs, nyq, order)

st.markdown("### Lowpass filtered z-axis data")
st.line_chart(df, x="Time (s)", y="filtered_z", x_label="Time (s)", y_label="Filtered Acceleration (z)")

st.markdown("### We can estimate step count by checking how many times the chart crosses up over a value on the y-axis")
with st.echo():
  def count_steps(values, c):
    steps = 0
    for i in range(1, len(values)):
      if (values[i-1] < c and values[i] > c):
        steps += 1
    return steps
  
  st.markdown(f"> #### Chart crosses up over y = 0 a total of _{count_steps(df["filtered_z"],0)} times_")


st.markdown("---")

st.markdown("## 2. Estimating step count by power spectrum made via FFT")
st.markdown("### Power spectrum")
st.line_chart(freqSpec, x="Frequencies", y="Magnitudes", x_label="Frequency (Hz)", y_label="Magnitude")

st.markdown("### We can estimate step count by looking at the highest magnitude frequency peak and multiplying by total length of data in seconds")
with st.echo():
  T = df["Time (s)"][len(df["Time (s)"]) - 1] - df["Time (s)"][0]
  peakFreq = frequencies[np.argmax(magnitudes)]
  freqSpecStepEstimate = T * peakFreq
  st.markdown(f"> #### {T:.2f} × {peakFreq:.2f} ≈ {freqSpecStepEstimate:.0f}")
st.markdown(f" ##### This is remarkably close to the previous estimate using lowpass filtering, which was {count_steps(df["filtered_z"],0)}")

st.markdown("---")
st.markdown("# Analyzing the GPS data")

st.markdown("## We can get the average velocity by taking the mean of all velocity data")
with st.echo():
  avg_velocity = gps["Velocity (m/s)"].mean()
  st.markdown(f"> #### The average velocity was {avg_velocity:0.3f} m/s")

st.markdown("## Then we get the total distance travelled using the haversine formula")
with st.echo():
  R = 6371000
  a = np.sin(np.radians(gps['Latitude (°)']).diff() / 2)**2 + np.cos(np.radians(gps['Latitude (°)'])) * np.cos(np.radians(gps['Latitude (°)']).shift()) * np.sin(np.radians(gps["Longitude (°)"]).diff() / 2)**2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  df['Distance (m)'] = R * c
  totalDist = df['Distance (m)'].sum()
  st.markdown(f"> #### Total distance travelled was {totalDist:0.1f} meters")

st.markdown("## And finally we can get the average length of step")
with st.echo():
  st.markdown(f"> #### Average lenght of step was {totalDist / freqSpecStepEstimate:0.2f} meters")

st.markdown("## Here is the route drawn on a map and a recap of calculated values")

col1, col2, = st.columns(2)
with col1:
  st_map = st_folium(m, width=900)

with col2:
  st.markdown("---")
  st.markdown(f"### Estimated step count using filtering: {count_steps(df["filtered_z"],0)}")
  st.markdown(f"### Estimated step count using power spectrum: {freqSpecStepEstimate:0.0f}")
  st.markdown(f"### Average velocity: {avg_velocity:0.3f} m/s")
  st.markdown(f"### Total distance travelled: {totalDist:0.1f} meters")
  st.markdown(f"### Average step length: {totalDist / freqSpecStepEstimate:0.1f} meters")