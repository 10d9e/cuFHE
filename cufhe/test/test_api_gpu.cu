/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Include these two files for GPU computing.
#include <include/cufhe_gpu.cuh>
using namespace cufhe;

#include <iostream>
using namespace std;

#include <vector>

void NandCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = 1 - in0.message_ * in1.message_;
}

void OrCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) > 0;
}

void AndCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = in0.message_ * in1.message_;
}

void XorCheck(Ptxt& out, const Ptxt& in0, const Ptxt& in1) {
  out.message_ = (in0.message_ + in1.message_) & 0x1;
}

int main() {
  const int num_gpus = 2; // Set the number of GPUs you want to use
  const uint32_t kNumTestsPerGPU = 16; // Number of tests per GPU
  const uint32_t kNumLevels = 4;

  // Create a vector to hold devices and streams for each GPU
  std::vector<int> devices(num_gpus);
  std::vector<Stream*> streams(num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i;
    streams[i] = new Stream[num_gpus];
    for (int j = 0; j < num_gpus; j++) {
      streams[i][j].Create();
    }
  }

  SetSeed(); // set random seed

  PriKey pri_key; // private key
  PubKey pub_key; // public key
  Ptxt* pt = new Ptxt[2 * kNumTests];
  Ctxt* ct = new Ctxt[2 * kNumTests];
  Synchronize();
  bool correct;

  cout<< "------ Key Generation ------" <<endl;
  KeyGen(pub_key, pri_key);
  // Alternatively ...
  // PriKeyGen(pri_key);
  // PubKeyGen(pub_key, pri_key);

  cout<< "------ Test Encryption/Decryption ------" <<endl;
  cout<< "Number of tests:\t" << kNumTests <<endl;
  correct = true;
  for (int i = 0; i < kNumTests; i ++) {
    pt[i].message_ = rand() % Ptxt::kPtxtSpace;
    Encrypt(ct[i], pt[i], pri_key);
    Decrypt(pt[kNumTests + i], ct[i], pri_key);
    if (pt[kNumTests + i].message_ != pt[i].message_) {
      correct = false;
      break;
    }
  }
  if (correct)
    cout<< "PASS" <<endl;
  else
    cout<< "FAIL" <<endl;

  // Test NAND gate (and other gates) on multiple GPUs
  std::cout << "------ Test NAND Gate ------" << std::endl;
  std::cout << "Number of tests per GPU:\t" << kNumTestsPerGPU << std::endl;
  bool correct = true;

  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(devices[i]);
    Initialize(pub_key); // essential for GPU computing

    for (int j = 0; j < kNumTestsPerGPU; j++) {
      // Create and encrypt data here
      Ptxt* pt = new Ptxt[2 * kNumTestsPerGPU];
      Ctxt* ct = new Ctxt[2 * kNumTestsPerGPU];
      for (int k = 0; k < 2 * kNumTestsPerGPU; k++) {
        pt[k].message_ = rand() % Ptxt::kPtxtSpace;
        Encrypt(ct[k], pt[k], pri_key);
      }
      Synchronize();
      
      // Perform gate operations on multiple GPUs here
      for (int k = 0; k < kNumTestsPerGPU; k++) {
        Nand(ct[k], ct[k], ct[k + kNumTestsPerGPU], streams[i][j]);
        Or(ct[k], ct[k], ct[k + kNumTestsPerGPU], streams[i][j]);
        And(ct[k], ct[k], ct[k + kNumTestsPerGPU], streams[i][j]);
        Xor(ct[k], ct[k], ct[k + kNumTestsPerGPU], streams[i][j]);
      }

      Synchronize();
      
      // Decrypt and check results here
      int cnt_failures = 0;
      for (int k = 0; k < kNumTestsPerGPU; k++) {
        NandCheck(pt[k], pt[k], pt[k + kNumTestsPerGPU]);
        OrCheck(pt[k], pt[k], pt[k + kNumTestsPerGPU]);
        AndCheck(pt[k], pt[k], pt[k + kNumTestsPerGPU]);
        XorCheck(pt[k], pt[k], pt[k + kNumTestsPerGPU]);
        Decrypt(pt[k + kNumTestsPerGPU], ct[k], pri_key);
        if (pt[k + kNumTestsPerGPU].message_ != pt[k].message_) {
          correct = false;
          cnt_failures += 1;
        }
      }
      if (!correct) {
        std::cout << "GPU " << i << " Test " << j << " FAIL:\t" << cnt_failures << "/" << kNumTestsPerGPU << std::endl;
      }
      
      // Clean up resources for this GPU
      delete[] ct;
      delete[] pt;
    }
    
    CleanUp(); // essential to clean and deallocate data
  }

  // Clean up streams and devices
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(devices[i]);
    for (int j = 0; j < num_gpus; j++) {
      streams[i][j].Destroy();
    }
    delete[] streams[i];
  }

  return 0;
}
