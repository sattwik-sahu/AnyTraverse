#!/usr/bin/bash

ngrok http 8501 \
    --oauth google \
    --oauth-allow-email agamdeep20@iiserb.ac.in \
    --oauth-allow-email sattwik21@iiserb.ac.in \
    --oauth-allow-email karthik23@iiserb.ac.in \
    --oauth-allow-email sujit@iiserb.ac.in
