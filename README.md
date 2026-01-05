Hello!

This repo is of an app that is hosted locally through streamlit. The purpose is to upload video footage of a person bowling, and use both frame-by-frame analysis, as well as total video analysis to determine if the action is deemed safe.
This uses a couple ML models to do so, all with specific Google Drive IDs, and they determine a risk factor.
In addition, if some joint angles are within a specific range, there will be some recommendations given, which you can run through again to see if the model thinks you have improved.
To make sure this works properly, use MediaPipe version 0.10.9, and use Python 3.10. Those were the versions this was created in, and will work in there.
In addition, you can link a fork of this repo to your own streamlit account to run properly, once you are sure you are using those versions!
