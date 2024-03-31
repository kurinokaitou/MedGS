import pickle
import tigre.algorithms as algs
from tigre.utilities.geometry import Geometry
import numpy as np

class ConeGeometry(Geometry):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data):
        Geometry.__init__(self)
        
        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source Detector      (m)
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


def reconstruct(projections, angles, geo, algo='FDK', iteration=50):
    print("Start pre-reconstruction using " + algo)
    if algo == 'OSSART':
        res = algs.ossart(projections, geo,angles, iteration,
                           **dict(blocksize=20,computel2=False))
    elif algo == 'SART':
        res = algs.sart(projections, geo, angles, iteration, 
                           **dict(computel2=False))
    elif algo == 'FDK':
        res = algs.fdk(projections, geo, angles, filter='ram_lak')
    else:
        assert False, "Unknown recontruction algorithm!"
    print("Finished pre pre-reconstruction")
    return res   
        
def main():
    with open("dataset/data.pickle", "rb") as handle:
        data = pickle.load(handle)
    geo = ConeGeometry(data)
    volume = reconstruct(data['train']["projections"],
                        data['train']["angles"],geo)
    print(volume.shape)


if __name__ == "__main__":
    main()
