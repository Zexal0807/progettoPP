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

int main()
{
    Point3D rayOrigin = {0.0, 0.0, 0.0};
    Point3D rayDirection = {0.0, 0.0, 2.0};

    Triangle triangle = {{-0.5, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, -1.0, 0.5}};
    // Triangle triangle = {{-0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, -0.2, 0.5}};

    bool intersezione = rayIntersectsTriangle(rayOrigin, rayDirection, triangle);

    if (intersezione)
        cout << "Il raggio interseca il triangolo." << endl;
    else
        cout << "Il raggio non interseca il triangolo." << endl;

    return 0;
}