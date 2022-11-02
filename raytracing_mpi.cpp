#include "minirt/minirt.h"
#include <mpi.h>
#include <cmath>
#include <iostream>

using namespace minirt;
using namespace std;

void initScene(Scene &scene) {
    Color red {1, 0.2, 0.2};
    Color blue {0.2, 0.2, 1};
    Color green {0.2, 1, 0.2};
    Color white {0.8, 0.8, 0.8};
    Color yellow {1, 1, 0.2};

    Material metallicRed {red, white, 50};
    Material mirrorBlack {Color {0.0}, Color {0.9}, 1000};
    Material matteWhite {Color {0.7}, Color {0.3}, 1};
    Material metallicYellow {yellow, white, 250};
    Material greenishGreen {green, 0.5, 0.5};

    Material transparentGreen {green, 0.8, 0.2};
    transparentGreen.makeTransparent(1.0, 1.03);
    Material transparentBlue {blue, 0.4, 0.6};
    transparentBlue.makeTransparent(0.9, 0.7);

    scene.addSphere(Sphere {{0, -2, 7}, 1, transparentBlue});
    scene.addSphere(Sphere {{-3, 2, 11}, 2, metallicRed});
    scene.addSphere(Sphere {{0, 2, 8}, 1, mirrorBlack});
    scene.addSphere(Sphere {{1.5, -0.5, 7}, 1, transparentGreen});
    scene.addSphere(Sphere {{-2, -1, 6}, 0.7, metallicYellow});
    scene.addSphere(Sphere {{2.2, 0.5, 9}, 1.2, matteWhite});
    scene.addSphere(Sphere {{4, -1, 10}, 0.7, metallicRed});

    scene.addLight(PointLight {{-15, 0, -15}, white});
    scene.addLight(PointLight {{1, 1, 0}, blue});
    scene.addLight(PointLight {{0, -10, 6}, red});

    scene.setBackground({0.05, 0.05, 0.08});
    scene.setAmbient({0.1, 0.1, 0.1});
    scene.setRecursionLimit(20);

    scene.setCamera(Camera {{0, 0, -20}, {0, 0, 0}});
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int viewPlaneResolutionX = (argc > 1 ? stoi(argv[1]) : 600);
    int viewPlaneResolutionY = (argc > 2 ? stoi(argv[2]) : 600);
    int numOfSamples = (argc > 3 ? stoi(argv[3]) : 1);
    string sceneFile = (argc > 4 ? argv[4] : "");

    Scene scene;
    if (sceneFile.empty()) {
        initScene(scene);
    } else {
        scene.loadFromFile(sceneFile);
    }

    const double backgroundSizeX = 4;
    const double backgroundSizeY = 4;
    const double backgroundDistance = 15;

    const double viewPlaneDistance = 5;
    const double viewPlaneSizeX = backgroundSizeX * viewPlaneDistance / backgroundDistance;
    const double viewPlaneSizeY = backgroundSizeY * viewPlaneDistance / backgroundDistance;

    ViewPlane viewPlane {viewPlaneResolutionX, viewPlaneResolutionY,
                         viewPlaneSizeX, viewPlaneSizeY, viewPlaneDistance};

    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Each process should compute a different piece of the image.
    int mySizeX = viewPlaneResolutionX / size;
    int mySizeY = viewPlaneResolutionY;

    // Each process stores its piece only.
    Image myPiece(mySizeX, mySizeY);
    
    
    double before = MPI_Wtime();

    for(int x = 0; x < mySizeX; x++) {
        for(int y = 0; y < mySizeY; y++) {
            const auto color = viewPlane.computePixel(scene, x + mySizeX * rank, y, numOfSamples);
            myPiece.set(x, y, color);
            } 
    }
    double after = MPI_Wtime();

    double time = after - before;
    
    cout << "Rank: " << rank << " Time: " << time << endl;

    double max_exc;
    MPI_Reduce(&time, &max_exc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) { cout << "Max Time:" <<  max_exc << endl;}
    
    MPI_Request send_request;
    MPI_Isend(myPiece.getData(), mySizeX * mySizeY * 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &send_request);

    if (rank == 0) {
        Image finalImage(viewPlaneResolutionX, viewPlaneResolutionY);

        for (int src_rank = 0; src_rank < size; src_rank++) {
            int srcSizeX = mySizeX;
            int srcSizeY = mySizeY;

            // Allocating temporary space to receive an image piece from src_rank process.
            Image srcPiece(srcSizeX, srcSizeY);
            
            // Receiving a piece.
            MPI_Recv(srcPiece.getData(), srcSizeX * srcSizeY * 3, MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int x = 0; x < mySizeX; x++) {
                for (int y = 0; y < mySizeY; y++) {
                    finalImage.set(x + src_rank * mySizeX, y, srcPiece.get(x, y));
	            }
            }
        }
        finalImage.saveJPEG("raytracing_" + to_string(size) + ".jpg");
    }
    
    // Finishing send operation.
    MPI_Wait(&send_request, MPI_STATUS_IGNORE);

    MPI_Finalize();

    return 0;
    
}