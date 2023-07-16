# Read for first time running

Deteksi wajah, usia, gender, dan emosi dirancang oleh tim Face Recognition magang Lab ISR TELKOM.

Tahap menjalankan kodingan:

1. Disarankan menggunakan Anaconda untuk meminimalkan packages yang perlu di install

2. Download atau clone repository ini

3. Download CMake dari website resminya [di sini](https://cmake.org/download/)

4. Install dlib dengan menggunakan perintah di bawah
(hanya berlaku untuk python versi 3.10.X)

`pip install dlib`

> Cek versi python, apabila belum versi 3.10.X sangat disarankan untuk update. Apabila dalam versi 3.7, 3.8, 3.9 dapat install dlib dengen menggunakan perintah:

### 3.7

`python -m pip install 'dlibwheel/dlib-19.22.99-cp37-cp37m-win_amd64.whl`

### 3.8

`python -m pip install 'dlibwheel/dlib-19.22.99-cp38-cp38m-win_amd64.whl`

### 3.9

`python -m pip install 'dlibwheel/dlib-19.22.99-cp39-cp39m-win_amd64.whl`

5. Setelah install dlib install requirements tambahan dengan mengetik perintah di bawah pada terminal:

`pip install -r requirements.txt`

6. Run main.py

`python 'src\main.py`
