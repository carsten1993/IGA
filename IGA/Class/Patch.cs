using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA
{
    class Patch
    {
        private List<List<List<Point3d>>> B, newB;
        private List<double> knotsCsi, knotsEta, knotsZeta;
        private int p, q, r, n, m, l, nel, nnp, nen;
        private List<List<int>> INN, IEN;

        public Patch() { }

        public Patch(List<List<List<Point3d>>> _B, List<double> _knotsCsi, List<double> _knotsEta, List<double> _knotsZeta, int _p, int _q, int _r, List<List<int>> _INN, List<List<int>> _IEN)
        {
            B = _B;
            newB = _B;
            knotsCsi = _knotsCsi;
            knotsEta = _knotsEta;
            knotsZeta = _knotsZeta;
            p = _p;
            q = _q;
            r = _r;
            n = knotsCsi.Count - (p + 1);
            m = knotsEta.Count - (q + 1);
            l = knotsZeta.Count - (r + 1);
            nel = (n - p) * (m - q) * (l - r);
            nnp = n * m * l;
            nen = (p + 1) * (q + 1) * (r + 1);        
            INN = _INN;
            IEN = _IEN;
        }

        public List<List<List<Point3d>>> GetControlPoints() { return B; }
        public List<List<List<Point3d>>> GetNewControlPoints() { return newB; }
        public (List<double>, List<double>, List<double>) GetKnotVectors() { return (knotsCsi, knotsEta, knotsZeta); }
        public (int, int, int) GetDegrees() { return (p, q, r); }
        public (int, int, int) GetNML() { return (n, m, l); }
        public (int, int, int) GetSize() { return (nel, nnp, nen); }
        public (List<List<int>>, List<List<int>>) GetINNIEN() { return (INN, IEN); }
        public string GetMeshInfo()
        {
            string info = "Total number of control points/shape functions: " + nnp +
                "\nNumber of control points/shape functions in the Csi, Eta and Zeta-direction: " + n + ", " + m + ", " + l +
                "\nNumber of local control points/shape functions: " + nen +
                "\nNumber of elements: " + nel +
                "\nDegrees in the Csi, Eta and Zeta-direction: " + p + ", " + q + ", " + r +
                "\nCsi knot vector: [";

            List<string> csi = new List<string>();
            List<string> eta = new List<string>();
            List<string> zeta = new List<string>();

            foreach(double d in knotsCsi)
            {
                info += " " + d.ToString();
            }
            info += " ]" + "\nEta knot vector: [";
            foreach (double d in knotsEta)
            {
                info += " " + d.ToString(); 
            }
            info += " ]" + "\nZeta knot vector: [";
            foreach (double d in knotsZeta)
            {
                info += " " + d.ToString();
            }
            info += " ]";

            /*"Total number of control points/shape functions: " + nnp +
                "\n Number of control points/shape functions in the Csi, Eta and Zeta-direction: " + n + ", " + m + ", " + l +
                "\n Number of local control points/shape functions: " + nnp +
                "\n Number of elements: " + nel +
                "\n Degrees in the Csi, Eta and Zeta-direction: " + p + ", " + q + ", " + r +
                "\n Csi knot vector: [" + csi + "]" +
                "\n Eta knot vector: [" + eta + "]" +
                "\n Zeta knot vector: [" + zeta + "]";*/

            return info;
        }
        public List<int> GetSurface1()
        {
            List<int> surface = new List<int>();

            for (int i = 1; i <= n * m; i++)
            {
                surface.Add(i);
            }

            return surface;
        }
        public List<int> GetSurface2()
        {
            List<int> surface = new List<int>();

            for (int k = 0; k < l; k++)
            {
                for (int i = 1; i <= n; i++)
                {
                    surface.Add(i + k * n * m);
                }
            }

            return surface;
        }
        public List<int> GetSurface3()
        {
            List<int> surface = new List<int>();

            for (int k = 0; k < l; k++)
            {
                for (int j = 1; j <= m; j++)
                {
                    surface.Add(j * n + k * n * m);
                }
            }

            return surface;
        }
        public List<int> GetSurface4()
        {
            List<int> surface = new List<int>();

            for (int k = 0; k < l; k++)
            {
                for (int i = 1; i <= n; i++)
                {
                    surface.Add(i + n * (m - 1) + k * n * m);
                }
            }

            return surface;
        }
        public List<int> GetSurface5()
        {
            List<int> surface = new List<int>();

            for (int k = 0; k < l; k++)
            {
                for (int j = 1; j <= m; j++)
                {
                    surface.Add(j * n - n + 1 + k * n * m);
                }
            }

            return surface;
        }
        public List<int> GetSurface6()
        {
            List<int> surface = new List<int>();

            for (int i = n * m * (l - 1) ; i <= n * m * l; i++)
            {
                surface.Add(i);
            }

            return surface;
        }

        public void SetNewControlPoints(Vector<double> d)
        {
            int id;
            for (int k = 0; k < l; k++)
            {
                for (int j = 0; j < m; j++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        id = k * n * m + j * n + i;
                        newB[k][j][i] += new Point3d(d[id * 3], d[id * 3 + 1], d[id * 3 + 2]);
                    }
                }
            }
        }
    }
}
