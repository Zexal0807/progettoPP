#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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

Point3D difference(Point3D a, Point3D b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Point3D crossProduct(Point3D &v1, Point3D &v2)
{
    return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x};
}

double dotProduct(Point3D &v1, Point3D &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

bool rayIntersectsTriangle(Point3D rayOrigin, Point3D rayVector,
                           Triangle inTriangle)
{
    const float EPSILON = 0.0000001;
    Point3D vertex0 = inTriangle.p1;
    Point3D vertex1 = inTriangle.p2;
    Point3D vertex2 = inTriangle.p3;

    Point3D edge1, edge2, h, s, q;
    double a, f, u, v;

    edge1 = difference(vertex1, vertex0);
    edge2 = difference(vertex2, vertex0);

    h = crossProduct(rayVector, edge2);
    a = dotProduct(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;

    f = 1.0 / a;
    s = difference(rayOrigin, vertex0);
    u = f * dotProduct(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    q = crossProduct(s, edge1);
    v = f * dotProduct(rayVector, q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double t = f * dotProduct(edge2, q);

    if (t > EPSILON)
        return true;
    return false;
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

bool rayIntersectsAnyTriangle(Point3D origin, Point3D dir,
                              vector<Triangle> triangles)
{
    for (const Triangle &triangle : triangles)
    {
        bool r = rayIntersectsTriangle(origin, dir, triangle);
        if (r)
        {
            return true;
        }
    }
    return false;
}

int main()
{

    ofstream oFile("out.txt");
    vector<Point3D> punti = readPoints("verts.csv");
    vector<Triangle> triangoli = readTriangles("meshes.csv", punti);

    Point3D rayOrigin = {0.0, 0.0, 0.0};

    for (const Point3D &punto : punti)
    {
        if (punto.x > 0 || punto.x <= 0)
            oFile << rayIntersectsAnyTriangle(rayOrigin, punto, triangoli) << endl;
    }
    oFile.close();

    return 0;
}