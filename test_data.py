import nibabel as nib
img = nib.load("data/sample_fastmri/sample.nii.gz")
vol = img.get_fdata()
print("shape:", vol.shape)
print("min/max:", vol.min(), vol.max())
