This is a stateless implementation of DPVO (and DPVO-SLAM) where multiple clients can call the server to get service. Currently, the client gives a path of the .MOV file. The model weights are stored locally right now (dpvo.pth).

1. `dpvo_stateless.py` &rarr; the server  
2. `dpvo_client.py` &rarr; client example

**First install dependencies** (requires `requirements.txt` which is already in the repo):  
To install dependencies, run the setup script:

```bash
chmod +x setup_dpvo.sh
./setup_dpvo.sh
```

**Change directory and activate venv**
```bash
cd DPVO
source .venv/bin/activate
```

**Run the server** in one terminal:

```bash
python dpvo_stateless.py
```

**Run the client** as follows:

```bash
python dpvo_client.py \
    --imagedir=movies/IMG_0492.MOV \
    --calib=calib/iphone.txt \
    --stride=5 \
    --plot \
    --opts LOOP_CLOSURE True \
    --name result_client_server
```

Currently, the server can store the point cloud, trajectory, camera poses, points, colors. Use colmap gui to visualize the points. The client does not receive any computed information apart from the metrics like fps.

Check the `README.md` for more details.

---

**TODO:**
1. Currently, the server returns the output to the client. Change this to store them somewhere for later access. [DONE]
2. Check for redundancy in storing the SLAM state in the server (extract/deserialize state). [DONE]
3. The setup file currently installs everything needed to train and test DPVO. This is not required since we only need inference. So need to write a minimalistic implementation.
4. Host model wieghts somewhere and call it from there, currently its taken from local.
5. Currently, slam.terminate() is called only after the server receives the last frame (denoted by timestamp = -1). Check if we need to call this after every frame/every few frames. [DONE] -- no need to do this explicitly (for now) since we are just sharing the last pose.
6. Store the outputs (pointcloud, etc.) on server side instead -- check where exactly to store them [DONE]
7. Testing (lots of it) [DONE] -- tested on iphone videos and euroc dataset.
8. If sending poses, timestamps, etc to client, then encode it before sending. --> Decided to just share last pose for now in the response (along with the timestamp).
9. Frequency of demo.py is ~13 hz and drops to ~2-3 hz for stateless implementation.
    - Identified which tensors need to be tranferred to CPU (memory buffers and patch graph network). Increases frequency by just ~0.5 hz.
    - If we store them on GPU too (for one client, it increases GPU usage by ~500 MB but increases slowly with number of frames processed), then frequency is > 10 hz.