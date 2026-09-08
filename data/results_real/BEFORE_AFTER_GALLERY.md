# Before / After Gallery

One complete case per degradation type: input, detected region, enhanced crop, and recovered text vs. ground truth.

## motion_blur  (`us_wts-lg-000016`, region: us)

![motion_blur](full_case_panels\motion_blur_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `` (char-acc 0.0, exact match: False)
- OCR after enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.866 -> 0.902
- SSIM before -> after: 0.822 -> 0.493  |  PSNR before -> after: 34.16 -> 21.55

## defocus_blur  (`us_wts-lg-000148`, region: us)

![defocus_blur](full_case_panels\defocus_blur_us_wts-lg-000148_FULL.png)

- Ground truth: `6TIX874`
- OCR before enhancement: `` (char-acc 0.0, exact match: False)
- OCR after enhancement: `` (char-acc 0.0, exact match: False)
- Detection IoU before -> after: 0.841 -> 0.85
- SSIM before -> after: 0.139 -> 0.189  |  PSNR before -> after: 12.96 -> 12.54

## low_res  (`us_wts-lg-000016`, region: us)

![low_res](full_case_panels\low_res_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `11111` (char-acc 0.0, exact match: False)
- OCR after enhancement: `0L1119` (char-acc 0.167, exact match: False)
- Detection IoU before -> after: 0.802 -> 0.802
- SSIM before -> after: 0.793 -> 0.55  |  PSNR before -> after: 32.56 -> 26.37

## noise  (`us_wts-lg-000010`, region: us)

![noise](full_case_panels\noise_us_wts-lg-000010_FULL.png)

- Ground truth: `HH9G5W`
- OCR before enhancement: `HH9G5W` (char-acc 1.0, exact match: True)
- OCR after enhancement: `HH8054` (char-acc 0.5, exact match: False)
- Detection IoU before -> after: 0.742 -> 0.713
- SSIM before -> after: 0.48 -> 0.069  |  PSNR before -> after: 24.86 -> 15.14

## jpeg_compression  (`us_wts-lg-000016`, region: us)

![jpeg_compression](full_case_panels\jpeg_compression_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `DLJC2Z` (char-acc 0.833, exact match: False)
- OCR after enhancement: `DLJC2Z` (char-acc 0.833, exact match: False)
- Detection IoU before -> after: 0.813 -> 0.763
- SSIM before -> after: 0.871 -> 0.601  |  PSNR before -> after: 35.16 -> 19.92

## under_exposure  (`us_wts-lg-000016`, region: us)

![under_exposure](full_case_panels\under_exposure_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `DL3C27` (char-acc 0.833, exact match: False)
- OCR after enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.843 -> 0.901
- SSIM before -> after: 0.551 -> 0.497  |  PSNR before -> after: 16.58 -> 24.48

## over_exposure  (`us_wts-lg-000016`, region: us)

![over_exposure](full_case_panels\over_exposure_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- OCR after enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.937 -> 0.92
- SSIM before -> after: 0.64 -> 0.138  |  PSNR before -> after: 8.13 -> 6.22

## perspective  (`us_wts-lg-000016`, region: us)

![perspective](full_case_panels\perspective_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- OCR after enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.642 -> 0.642
- SSIM before -> after: 0.6 -> 0.539  |  PSNR before -> after: 28.22 -> 14.89

## combined_hard  (`br_OYJ9557`, region: br)

![combined_hard](full_case_panels\combined_hard_br_OYJ9557_FULL.png)

- Ground truth: `OYJ9557`
- OCR before enhancement: `DYJ9557` (char-acc 0.857, exact match: False)
- OCR after enhancement: `OYJ9557` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.974 -> 0.974
- SSIM before -> after: 0.524 -> 0.241  |  PSNR before -> after: 21.88 -> 12.84

## video_multiframe  (`us_wts-lg-000016`, region: us)

![video_multiframe](full_case_panels\video_multiframe_us_wts-lg-000016_FULL.png)

- Ground truth: `DL3C2Z`
- OCR before enhancement: `DL3C27` (char-acc 0.833, exact match: False)
- OCR after enhancement: `DL3C2Z` (char-acc 1.0, exact match: True)
- Detection IoU before -> after: 0.955 -> 0.955
- SSIM before -> after: 0.428 -> 0.315  |  PSNR before -> after: 27.66 -> 18.55
