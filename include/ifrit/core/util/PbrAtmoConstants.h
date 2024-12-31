#pragma once
#include "gcem.hpp"
#include <array>

// Code Modified from:
/**
 * Copyright (c) 2017 Eric Bruneton
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace Ifrit::Core::Util::PbrAtmoConstants {
constexpr double m = 1.0;
constexpr double m2 = 1.0;
constexpr double km = 1000.0;

constexpr int kLambdaMin = 360;
constexpr int kLambdaMax = 830;
constexpr double kSolarIrradiance[48] = {
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887,  1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685,  1.8931,  1.85149, 1.8504,  1.8341,  1.8345,  1.8147,  1.78158,
    1.7533,  1.6965,  1.68194, 1.64654, 1.6048,  1.52143, 1.55622, 1.5113,
    1.474,   1.4482,  1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758,
    1.2367,  1.2082,  1.18737, 1.14683, 1.12362, 1.1058,  1.07124, 1.04992};

constexpr double kRayleigh = 1.24062e-6 / m;
constexpr double kRayleighScaleHeight = 8000.0 * m;
constexpr double kMieScaleHeight = 1200.0 * m;
constexpr double kMieAngstromAlpha = 0.0;
constexpr double kMieAngstromBeta = 5.328e-3;
constexpr double kMieSingleScatteringAlbedo = 0.9;
constexpr double kMiePhaseFunctionG = 0.8;

constexpr double kLambdaR = 680.0;
constexpr double kLambdaG = 550.0;
constexpr double kLambdaB = 440.0;

constexpr double kOzoneCrossSection[48] = {
    1.18e-27,  2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25,  1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25,   4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25,  3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26,  4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27};

constexpr double kDobsonUnit = 2.687e20 / m2;
constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / (15.0 * km);

constexpr uint32_t kSpecSize = (kLambdaMax - kLambdaMin) / 10 + 1;
consteval std::array<double, kSpecSize> getSolarIrradiance() {
  std::array<double, kSpecSize> solarIrradiance;
  for (int i = 0; i < solarIrradiance.size(); ++i) {
    solarIrradiance[i] = kSolarIrradiance[i];
  }
  return solarIrradiance;
}

consteval std::array<float, kSpecSize> getRayleighScattering() {
  std::array<float, kSpecSize> rayleighScattering;
  for (int i = 0; i < rayleighScattering.size(); ++i) {
    double lambda = static_cast<double>(kLambdaMin + i * 10) * 1e-3;
    float rayleight = static_cast<float>(kRayleigh * gcem::pow(lambda, -4.0));
    rayleighScattering[i] = rayleight;
  }
  return rayleighScattering;
}

consteval std::array<float, kSpecSize> getMieScattering() {
  std::array<float, kSpecSize> mieScattering;
  for (int i = 0; i < mieScattering.size(); ++i) {
    double lambda = static_cast<float>(i) * 1e-3f;
    double mie = kMieAngstromBeta / kMieScaleHeight *
                gcem::pow(lambda, -kMieAngstromAlpha);
    mieScattering[i] = static_cast<float>(mie * kMieSingleScatteringAlbedo);
  }
  return mieScattering;
}

consteval std::array<float, kSpecSize> getMieExtinction() {
  std::array<float, kSpecSize> mieExtinction;
  for (int i = 0; i < mieExtinction.size(); ++i) {
    double lambda = static_cast<double>(i) * 1e-3;
    float mie = static_cast<float>( kMieAngstromBeta / kMieScaleHeight *
                gcem::pow(lambda, -kMieAngstromAlpha));
    mieExtinction[i] = mie;
  }
  return mieExtinction;
}

consteval std::array<float, kSpecSize> getAbsorptionExtinction() {
  std::array<float, kSpecSize> absorptionExtinction;
  for (int i = 0; i < absorptionExtinction.size(); ++i) {
    double lambda = static_cast<double>(i) * 1e-3;
    float val = static_cast<float>(kMaxOzoneNumberDensity * kOzoneCrossSection[i]);
    absorptionExtinction[i] = val;
  }
  return absorptionExtinction;
}

} // namespace Ifrit::Core::Util::PbrAtmoConstants