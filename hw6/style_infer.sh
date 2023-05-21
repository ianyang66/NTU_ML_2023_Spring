# Compress the images.
# stylegan2_pytorch --num_workers=8
python inference.py
cd ./submission_stylegan2
tar -zcf ../submission_stylegan2-40000.tgz *.jpg
cd ..
