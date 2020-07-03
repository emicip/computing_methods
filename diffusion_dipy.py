from os.path import expanduser, join
from os import listdir
import fnmatch
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere, small_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data, load_nifti
import dipy.denoise.pca_noise_estimate
from dipy.denoise.localpca import localpca
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)                             
from dipy.workflows.align import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.direction import peaks_from_model
import dipy.tracking.stopping_criterion
from dipy.viz import window, actor, colormap, has_fury, regtools
from dipy.reconst.shm import CsaOdfModel
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk

import nibabel as nib
import matplotlib.pyplot as plt

def load_data (PATH,diffusion_file_name,diffusion_bvalue,diffusion_bvec,brain_data):
    # Function responsible to load diffusion and anatomical data. 
    home = expanduser(PATH)

    for file in listdir(home):
        if fnmatch.fnmatch(file, diffusion_file_name):
            fdwi = join(home, file)
        elif fnmatch.fnmatch(file, diffusion_bvalue):
            fbval = join(home, file)
        elif fnmatch.fnmatch(file, diffusion_bvec):
            fbvec = join(home, file)
        elif fnmatch.fnmatch(file, brain_data):
            brain = join(home, file)

    diff, affine_diff, img = load_nifti(fdwi,return_img=True)
    brain,affine_brain=load_nifti(brain)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    return diff, affine_diff, brain,affine_brain, gtab, img

def denoise (diff,gtab,affine_diff):
    # PCA-based denoising algorithms are effective denoising methods because they explore the redundancy of the multi-dimensional information of diffusion-weighted datasets.

    sigma = dipy.denoise.pca_noise_estimate.pca_noise_estimate(diff, gtab, correct_bias=True, smooth=3)
    den = localpca(diff, sigma, tau_factor=2.3, patch_radius=2)
    nib.save(nib.Nifti1Image(den, affine_diff), 'denoised.nii.gz')
    return den

def registration(diff,affine_diff,anat,affine_anat):
   #Affine trasformation beetween diffuson and anatomical data 
    static = np.squeeze(diff)[..., 0]
    static_grid2world = affine_diff

    moving = anat
    moving_grid2world = affine_anat

    identity = np.eye(4)
    affine_map = AffineMap(identity,
                        static.shape, static_grid2world,
                        moving.shape, moving_grid2world)
    resampled = affine_map.transform(moving)
    regtools.overlay_slices(static, resampled, None, 0,
                            "Static", "Moving", "resampled_0.png")
    regtools.overlay_slices(static, resampled, None, 1,
                            "Static", "Moving", "resampled_1.png")
    regtools.overlay_slices(static, resampled, None, 2,
                            "Static", "Moving", "resampled_2.png")

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                        moving, moving_grid2world)

    transformed = c_of_mass.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_com_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_com_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_com_2.png")

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    factors = [4, 2, 1]
    sigmas = [3.0, 1.0, 0.0]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                static_grid2world, moving_grid2world,
                                starting_affine=starting_affine)

    transformed = translation.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_trans_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_trans_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_trans_2.png")

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigid.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_rigid_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_rigid_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_rigid_2.png")      


    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    transformed = affine.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_affine_2.png")

    inverse_map = AffineMap(starting_affine,
                        static.shape, static_grid2world,
                        moving.shape, moving_grid2world)
    resampled_inverse = inverse_map.transform_inverse(transformed,resample_only=True)
    nib.save(nib.Nifti1Image(resampled_inverse, affine_diff), 'brain.coreg.nii.gz')
    return transformed

def segmentation (brain,affine_brain,diff,affine_diff):
    # Segmentation of brain into three principal tissue components: white matter, gray matter, cerebrospinal fluid

    t1=registration(diff,affine_diff,brain,affine_brain)
    nclass = 3
    beta = 0.1
    hmrf = TissueClassifierHMRF()
    _,final_segmentation = hmrf.classify(t1, nclass, beta)

    nib.save(nib.Nifti1Image(final_segmentation, affine_diff), 'segmentation.nii.gz')
    return final_segmentation

def tractography(brain,affine_brain,labels,diff,affine_diff, gtab,img):
    # Tractography reconstruction based on EuDX determinist algorithm

    labels=segmentation (brain,affine_brain, diff,affine_diff)
    white_matter = (labels == 3)

    csa_model = CsaOdfModel(gtab, sh_order=2)
    csa_peaks = peaks_from_model(csa_model, diff, default_sphere,
                                relative_peak_threshold=.8,
                                min_separation_angle=45, mask=white_matter)

    stopping_criterion = dipy.tracking.stopping_criterion.ThresholdStoppingCriterion(csa_peaks.gfa, .25)
    seeds = utils.seeds_from_mask(white_matter, affine_diff, density=1)
    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                        affine=affine_diff, step_size=.5)
    streamlines = Streamlines(streamlines_generator)
    if has_fury:
        color = colormap.line_colors(streamlines)

        streamlines_actor = actor.line(streamlines, color)
        r = window.Renderer()
        r.add(streamlines_actor)
        window.record(r, out_path='tractogram.png', size=(800, 800))
        window.show(r)
    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    save_trk(sft, "tractogram.trk", streamlines)
    return streamlines

if __name__ == "__main__":
    pass
