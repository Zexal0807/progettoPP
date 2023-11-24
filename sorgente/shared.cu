#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#define SH_SIZE 512

using namespace std;

struct Point3D
{
    double x, y, z;
};

struct Triangle
{
    Point3D p1;
    Point3D p2;
    Point3D p3;
};

__device__ 
Point3D difference(Point3D a, Point3D b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ 
Point3D crossProduct(Point3D &v1, Point3D &v2)
{
    return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x};
}

__device__ 
double dotProduct(Point3D &v1, Point3D &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ 
bool rayIntersectsTriangle(Point3D rayOrigin, Point3D rayVector,
                           Triangle inTriangle)
{
    const float EPSILON = 0.00001;
    Point3D vertex0 = inTriangle.p1;
    Point3D vertex1 = inTriangle.p2;
    Point3D vertex2 = inTriangle.p3;

    Point3D edge1, edge2, pvec, tvec, qvec;
    double det, f, u, v;

    edge1 = difference(vertex1, vertex0);
    edge2 = difference(vertex2, vertex0);

    pvec = crossProduct(rayVector, edge2);
    det = dotProduct(edge1, pvec);

    if (abs(det) < EPSILON)
    {
        return false;
    }

    tvec = difference(rayOrigin, vertex0);

    f = 1.0 / det;
    u = f * dotProduct(tvec, pvec);

    if (u < 0.0 + EPSILON || u > 1.0 - EPSILON)
    {
        return false;
    }

    qvec = crossProduct(tvec, edge1);

    v = dotProduct(rayVector, qvec);
    v = f * v;

    if (v < 0.0 + EPSILON || u + v > 1.0 - EPSILON)
    {
        return false;
    }

    double t = f * dotProduct(edge2, qvec);

    if (t >= EPSILON && t <= 1.0 + EPSILON)
    {
        return true;
    }
    return false;
}

__global__
void rayIntersectsAnyTrianglesKernel(
	Point3D rayOrigin, 
	Point3D *ps,
	Triangle *ts,
	int Nt,
	bool *result
){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	Point3D point = ps[idx];
	
	Point3D dir = point;

	__shared__ Triangle sh_triangle[SH_SIZE];
	
	result[idx] = true;
	
	for(int m = 0; m < Nt/SH_SIZE; m++){
		
		if(idx < SH_SIZE){
			sh_triangle[idx] = ts[m * SH_SIZE + idx];
		}
		
		__syncthreads();
		
		for(int i = 0; i < SH_SIZE; i++){
			Triangle t = sh_triangle[i];
			
			bool r = rayIntersectsTriangle(rayOrigin, dir, t);
			if(r){
				result[idx] = false;
			}
		}
		__syncthreads();
		
	}
}

vector<Point3D> readPoints(string filename)
{
    vector<Point3D> punti;
    string linea;
    ifstream file(filename);

    if (file.is_open())
    {
        while (getline(file, linea))
        {
            Point3D punto;
            stringstream ss(linea);
            string valore;
            if (getline(ss, valore, ','))
            {
                punto.x = stod(valore);
            }
            if (getline(ss, valore, ','))
            {
                punto.y = stod(valore);
            }
            if (getline(ss, valore))
            {
                punto.z = stod(valore);
            }
            punti.push_back(punto);
        }

        file.close();
    }
    return punti;
}

vector<Triangle> readTriangles(string filename, vector<Point3D> punti)
{
    vector<Triangle> triangoli;
    string linea;
    ifstream file(filename);

    if (file.is_open())
    {
        while (getline(file, linea))
        {
            Triangle t;
            stringstream ss(linea);
            string valore;

            int index = 0;

            if (getline(ss, valore, ','))
            {
                index = (int)stod(valore);
                t.p1 = punti[index];
            }
            if (getline(ss, valore, ','))
            {
                index = (int)stod(valore);
                t.p2 = punti[index];
            }
            if (getline(ss, valore))
            {
                index = (int)stod(valore);
                t.p3 = punti[index];
            }
            triangoli.push_back(t);
        }

        file.close();
    }
    return triangoli;
}

int main()
{
    vector<Point3D> punti = readPoints("rotated_verts.csv");
    vector<Triangle> triangoli = readTriangles("meshes.csv", punti);

	int Np = punti.size();
	int Nt = triangoli.size();

	Point3D rayOrigin = {0.0, 0.0, 0.0};
	
	Point3D *h_punti = new Point3D[Np];
	for(int i = 0; i < Np; i++){
		h_punti[i] = punti[i];
	}
	
	Triangle *h_triangoli = new Triangle[Nt];
	for(int i = 0; i < Nt; i++){
		h_triangoli[i] = triangoli[i];
	}
	
	bool *result = new bool[Np];
	
	Point3D *d_points;
	Triangle *d_triangles;
	bool *d_result;
	
	// DEVICE MEMORY ALLOCATION
	cudaMalloc(&d_points, Np * sizeof(Point3D) );
	cudaMalloc(&d_triangles, Nt * sizeof(Triangle) );
	cudaMalloc(&d_result, Np * sizeof(bool) );
	
	// COPY DATA FROM HOST TO DEVIE
	cudaMemcpy( d_points, h_punti, Np * sizeof(Point3D), cudaMemcpyHostToDevice);
	cudaMemcpy( d_triangles, h_triangoli, Nt * sizeof(Triangle), cudaMemcpyHostToDevice);
	
	// DEVICE INIT
	dim3 DimGrid(Np/256, 1, 1);
	if (Np % 256)
		DimGrid.x++;
	dim3 DimBlock(256, 1, 1);
	
	// DEVICE EXECUTION
	rayIntersectsAnyTrianglesKernel<<<DimGrid, DimBlock>>>(rayOrigin, d_points, d_triangles, Nt, d_result);
	
	// COPY DATA FROM DEVICE TO HOST
	cudaMemcpy(result, d_result, Np * sizeof(bool), cudaMemcpyDeviceToHost);
	
	// DEVICE MEMORY DEALLOCATION
	cudaFree( d_points );
	cudaFree( d_triangles );
	cudaFree( d_result );
	
	// PRINT RESULT
	ofstream oFile("out.txt");
	for(int i = 0; i < Np; i++) {
		oFile << result[i] << endl;
	}	
    oFile.close();
    return 0;
}