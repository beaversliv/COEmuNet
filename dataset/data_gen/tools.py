import os
import numpy              as np
import magritte.core      as magritte
from magritte.core        import ImageType, ImagePointPosition  # Image type, point position
from astropy              import units,constants              # Unit conversions
import scipy as sp
# from scipy.interpolate    import griddata           # Grid interpolation
def intensity(model):
    model.compute_spectral_discretisation ()
    model.compute_inverse_line_widths     ()
    model.compute_LTE_level_populations   ()

    fcen = model.lines.lineProducingSpecies[0].linedata.frequency[0]
    vpix = 300   # velocity pixel size [m/s] 
    dd   = vpix * (model.parameters.nfreqs()-1)/2 / magritte.CC
    fmin = fcen - fcen*dd
    fmax = fcen + fcen*dd
    # frequencies = np.linspace(fmin,fmax,n_quad,dtype=float64)

    # start computing Intensity
    model.compute_spectral_discretisation (fmin, fmax)
    # along the first ray
    model.compute_image (0)

    image_nr = -1
    zoom     = 1.0
    npix_x = 256
    npix_y = 256
    method     = 'nearest'
    # Extract data of last image
    imx = np.array(model.images[image_nr].ImX)
    imy = np.array(model.images[image_nr].ImY)
    imI = np.array(model.images[image_nr].I)

    # Workaround for model images
    if (model.images[image_nr].imagePointPosition == ImagePointPosition.AllModelPoints):
    # if (False):
        # Filter imaging data originating from boundary points
        bdy_indices = np.array(model.geometry.boundary.boundary2point)
        imx = np.delete(imx, bdy_indices)
        imy = np.delete(imy, bdy_indices)
        imI = np.delete(imI, bdy_indices, axis=0)

    # Extract the number of frequency bins
    nfreqs = model.images[image_nr].nfreqs

    # Set image boundaries
    deltax = (np.max(imx) - np.min(imx))/zoom
    midx = (np.max(imx) + np.min(imx))/2.0
    deltay = (np.max(imy) - np.min(imy))/zoom
    midy = (np.max(imy) + np.min(imy))/2.0

    x_min, x_max = midx - deltax/2.0, midx + deltax/2.0
    y_min, y_max = midy - deltay/2.0, midy + deltay/2.0

    # Create image grid values
    xs = np.linspace(x_min, x_max, npix_x)
    ys = np.linspace(y_min, y_max, npix_y)

    # Extract the spectral / velocity data
    freqs = np.array(model.images[image_nr].freqs)
    f_ij  = np.mean(freqs)
    # velos = (freqs - f_ij) / f_ij * constants.c.to(v_unit).value

    # Interpolate the scattered data to an image (regular grid)
    Is = np.zeros((nfreqs))
    zs = np.zeros((nfreqs, npix_x, npix_y))
    for f in range(nfreqs):
        # Nearest neighbor interpolate scattered image data
        zs[f] = sp.interpolate.griddata(
            (imx, imy),
            imI[:,f],
            (xs[None,:], ys[:,None]),
            method=method,
            fill_value = 0.0 #for non-nearest neighbor interpolation, otherwise the ceil/floor functions will complain
        )
        Is[f] = np.sum(zs[f])
    Is = Is / np.max(Is)

    # Put zero/negative values to the smallest positive value
    zs[zs<=0.0] = np.min(zs[zs>0.0])
    # Put nan values to smallest positive value
    zs[np.isnan(zs)] = np.min(zs[zs>0.0])
    return xs,ys,zs,freqs

def model_find():
    '''
    Absolute path of all original magritte models
    xxxx.hdf5
    '''
    model_files = []
    path = "/home/dc-su2/rds/rds-dirac-dp012/dc-su2/AMRVAC_3D/"
    for model_dir in os.listdir(path):
        # model_dir is modelxxx
        model_path = os.path.join(path,model_dir)
        for file_dir in os.listdir(model_path):
            #file_dir is 0789
            current_path = os.path.join(model_path,file_dir)
            model_file = os.path.join(current_path,f'{file_dir}.hdf5')

            model_files.append(model_file) 
    return model_files
def path_rotations(model_files,dir='Rotation_Dataset',type_='Original'):
    '''
    Generate path (folders) for rotated model files
    ---------
    input:
        model_files: list, paths for old magritte model files
    return:
        r_model_files: list, paths for rotated magritte model files
        dataset files: list, paths for extracted dataset(input,zs,xs,ys,nu) corresponding with rotated magritte model
    '''
    r_model_files = []
    dataset_files = []
    for model_file in model_files:
        # split path according to '/'
        dir_list = model_file.split('/')
        # change AMRVAC_3D to Rotation_Dataset
        dir_list[7] = dir
        # list, splited elements 
        dir_list.insert(8,type_)
        r_model_file = ('/').join(dir_list)

        dataset_file = f'{os.path.split(r_model_file)[0]}/dataset.hdf5'
        r_model_files.append(r_model_file)
        dataset_files.append(dataset_file)
    # creat folders for rotated magritte model
    for idx,r_model_file in enumerate(r_model_files):
        model_dir = os.path.split(r_model_file)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    return r_model_files,dataset_files
    
def modify_path_and_ensure_directory(original_path):
    # Insert 'grid32' after 'cnn'
    parts = original_path.split('/')
    insert_index = parts.index('cnn') + 1
    parts.insert(insert_index, 'grid32')
    modified_path = '/'.join(parts)
    
    # Extract the directory path to check/create
    directory_path = '/'.join(parts[:insert_index + 3])
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    return modified_path

def plot_cube_2D(cube):
    """
    Plot a slice along the third axis through a 3D cube.
    """
    vmin = cube.min().item()
    vmax = cube.max().item()
    def plot(z):
        plt.figure(dpi=150)
        plt.imshow(cube[:,:,z].T.data, vmin=vmin, vmax=vmax, origin='lower')
    return interact(plot, z=(0,cube.shape[2]-1))

# try:
#         magritte.Model(model_file)
#     except RuntimeError:
#     # re-write some parameters for compating with new Magritte environment
#         with h5.File(model_file, 'a') as file:
#             # Setting parameters.
#             file.attrs['use_scattering'] = 'false'
#             file.attrs['hnrays'] = 6
#             file.attrs['nlines'] = 1
#             file.attrs['nfreqs'] = 51
#         print(f'finish re-wirte: {model_file}')