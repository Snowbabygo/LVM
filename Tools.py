import numpy as np
import skimage
import numpy as np
import cv2 as cv

def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):

    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """

    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }

    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)

    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    rgb = np.clip(rgb, 0, 1)

    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)


class RandomRotate:
    '''takes in image of size (npix, npix, nchannel), flips l/r and or u/d, then rotates (0,360)'''

    def __call__(self, image):

        if np.random.randint(0, 2) == 1:
            image = np.flip(image, axis=0)
        if np.random.randint(0, 2) == 1:
            image = np.flip(image, axis=1)

        return skimage.transform.rotate(image, float(360 * np.random.rand(1)))


class JitterCrop:
    '''takes in image of size (npix, npix, nchannel),
    jitters by uniformly drawn (-jitter_lim, jitter_lim),
    and returns (outdim, outdim, nchannel) central pixels'''

    def __init__(self, outdim=96, jitter_lim=7):
        self.outdim = outdim
        self.jitter_lim = jitter_lim

    def __call__(self, image):
        if self.jitter_lim:
            center_x = image.shape[0] // 2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim + 1, 1))
            center_y = image.shape[0] // 2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim + 1, 1))
        else:
            center_x = image.shape[0] // 2
            center_y = image.shape[0] // 2
        offset = self.outdim // 2
        # print("crop",image[(center_x-offset):(center_x+offset), (center_y-offset):(center_y+offset)].shape)
        return image[(center_x - offset):(center_x + offset), (center_y - offset):(center_y + offset)]


class SizeScale:
    '''takes in image of size (npix, npix, nchannel), and scales the size larger or smaller
    anti-aliasing should probably be enabled when down-sizing images to avoid aliasing artifacts

    This augmentation changes the number of pixels in an image. After sizescale, we still need enough
    pixels to allow for jitter crop to not run out of bounds. Therefore,

    scale_min >= (outdim + 2*jitter_lim)/indim

    if outdim = 96, and indim=152 and jitter_lim = 7, then scale_min >= 0.73.

    When using sizescale, there is a small possibility that one corner of the image can be set to 0 in randomrotate,
    then the image can be scaled smaller, and if the image is jittered by near the maximum allowed value, that these
    0s will remain in a corner of the final image. Adding Gaussian noise after all the other augmentations should
    remove any negative effects of this small 0 patch.
    '''

    def __init__(self, scale_min=0.7, scale_max=3):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, image):
        scalei = np.random.uniform(self.scale_min, self.scale_max)
        output_shape = (int(image.shape[0] * scalei), int(image.shape[1] * scalei), image.shape[2])

        # print("size",skimage.transform.resize(image, output_shape).shape)

        return skimage.transform.resize(image, output_shape)


class GaussianNoise:
    '''adds Gaussian noise consistent from distribution fit to decals south \sigma_{pix,coadd}
    (see https://www.legacysurvey.org/dr9/nea/) as measured from 43e6 samples with zmag<20.

    Images already have noise level when observed on sky, so we do not want
    to add a total amount of noise, we only want to augment images by the
    difference in noise levels between various objects in the survey.

    1/sigma_pix^2 = psf_depth * [4pi (psf_size/2.3548/pixelsize)^2],
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec,
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma

    noise in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured noise distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):

    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(0.001094, 0.013), Lognormal fit (shape, loc, scale)=(0.2264926, -0.0006735, 0.0037602)
    # r: (min, max)=(0.001094, 0.018), Lognormal fit (shape, loc, scale)=(0.2431146, -0.0023663, 0.0067417)
    # z: (min, max)=(0.001094, 0.061), Lognormal fit (shape, loc, scale)=(0.1334844, -0.0143416, 0.0260779)
    '''

    def __init__(self, scaling=[1.], mean=0, im_dim=152, im_ch=3, decals=True, uniform=False):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image):

        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final ** 2 - self.sigma_true ** 2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)

        a = np.zeros(shape=(image.shape))

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                a[:, :, i] = image[:, :, i] + np.random.normal(self.mean, self.sigma_augment[i],
                                                               size=(self.im_dim, self.im_dim))
            else:
                a[:, :, i] = image[:, :, i]

        return a


class GaussianBlur:
    '''adds Gaussian PSF blur consistent from distribution fit to decals psf_size
    from sweep catalogues as measured from 2e6 spectroscopic samples.
    Images have already been filtered by PSF when observed on sky, so we do not want
    to smooth images using a total smoothing, we only want to augment images by the
    difference in smoothings between various objects in the survey.

    sigma = psf_size / pixelsize / 2.3548,
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec,
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma

    PSF in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured PSF distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):

    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(1.3233109, 5), Lognormal fit (shape, loc, scale)=(0.2109966, 1.0807153, 1.3153171)
    # r: (min, max)=(1.2667341, 4.5), Lognormal fit (shape, loc, scale)=(0.3008485, 1.2394326, 0.9164757)
    # z: (min, max)=(1.2126263, 4.25), Lognormal fit (shape, loc, scale)=(0.3471172, 1.1928363, 0.8233702)
    '''

    def __init__(self, scaling=[1.], im_dim=152, im_ch=3, decals=True, uniform=False):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5., 4.5, 4.25])

    def __call__(self, image):
        # noise in channels is uncorrelated, as images taken at different times/telescopes

        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final ** 2 - self.sigma_true ** 2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)

        a = np.zeros(shape=(image.shape))

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                a[:, :, i] = skimage.filters.gaussian(image[:, :, i], sigma=self.sigma_augment[i], mode='reflect')
            else:
                a[:, :, i] = image[:, :, i]
        return a



def DESI_find_Contour(image):
    '''takes in image of size (npix, npix,nchannel)'''

    data = dr2_rgb(np.transpose(image, (2, 1, 0)), ['g', 'r', 'z'])
    gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    s = gray.shape[1]

    # normalzation
    gray = np.uint8((gray - np.min(gray)) * 100 / (np.max(gray) - np.min(gray)))

    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 找到中心位置的框
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if x <= 0 or y <= 0:
            pass
        elif x + w > s or y + h > s:
            pass
        else:
            if x < s / 2 and x + w > s / 2 and y < s / 2 and y + h > s / 2:
                # 确定 crop_size，因为沿着边，所以+10
                crop_size = max(int(x + w - s / 2), int(y + h - s / 2), int(s / 2 - x), int(s / 2 - y)) + 10
                if crop_size > int(s / 2):
                    crop_size = int(s / 2)
                return int(s / 2) - crop_size, int(s / 2) + crop_size, int(s / 2) - crop_size, int(s / 2) + crop_size

    return 0, s, 0, s
