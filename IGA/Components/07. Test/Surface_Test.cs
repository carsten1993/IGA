using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Surface_Test : GH_Component
    {

        public Surface_Test()
          : base("Surface Test", "surface_test",
              "Testing if points are within the surface",
              "IGA", "07. Test")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Control points", "B", "Control points as a list", GH_ParamAccess.list);
            pManager.AddNumberParameter("Knot vector csi", "knotsCsi", "Knot vector csi", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree csi", "p", "Degree csi", GH_ParamAccess.item, 0);
            pManager.AddIntegerParameter("Index csi", "i", "Index csi", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("Csi", "csi", "Csi", GH_ParamAccess.item, 0);
            pManager.AddNumberParameter("Knot vector eta", "knotsEta", "Knot vector eta", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree eta", "q", "Degree eta", GH_ParamAccess.item, 0);
            pManager.AddIntegerParameter("Index eta", "j", "Index eta", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("Eta", "eta", "Eta", GH_ParamAccess.item, 0);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Point", "point", "Point on surface for the given csi and eta", GH_ParamAccess.item);
            pManager.AddPointParameter("Derivative csi", "dPoint_csi", "The derivative of the surface at the point given by csi", GH_ParamAccess.item);
            pManager.AddPointParameter("Derivative eta", "dPoint_eta", "The derivative of the surface at the point given by eta", GH_ParamAccess.item);
            pManager.AddNumberParameter("Basis function csi", "N", "The basis function w.r.t. csi, given the degree, index and csi", GH_ParamAccess.item);
            pManager.AddNumberParameter("Derivative basis function csi", "dN_dCsi", "The derivative of the basis function w.r.t. csi", GH_ParamAccess.item);
            pManager.AddNumberParameter("Basis function eta", "M", "The basis function w.r.t. eta, given the degree, index and eta", GH_ParamAccess.item);
            pManager.AddNumberParameter("Derivative basis function eta", "dM_dEta", "The derivative of the basis function w.r.t. eta", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<Point3d> controlPoints = new List<Point3d>();
            List<double> knotsCsi = new List<double>();
            int p = 0;
            int i = 1;
            double csi = 0;
            List<double> knotsEta = new List<double>();
            int q = 0;
            int j = 1;
            double eta = 0;

            if (!DA.GetDataList(0, controlPoints)) return;
            if (!DA.GetDataList(1, knotsCsi)) return;
            if (!DA.GetData(2, ref p)) return;
            if (!DA.GetData(3, ref i)) return;
            if (!DA.GetData(4, ref csi)) return;
            if (!DA.GetDataList(5, knotsEta)) return;
            if (!DA.GetData(6, ref q)) return;
            if (!DA.GetData(7, ref j)) return;
            if (!DA.GetData(8, ref eta)) return;

            int n = knotsCsi.Count - p - 1;
            int m = knotsEta.Count - q - 1;

            if (i > n)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Index 1 can not be greater than or equal to the number of knots minus the degree");
                return;
            }
            if (j > m)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Index 2 can not be greater than or equal to the number of knots minus the degree");
                return;
            }
            if (n*m != controlPoints.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Number of control points does not match the number of basis function for the given degrees");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<List<Point3d>> B = CreateControlPointList(controlPoints, n, m);
            Point3d point = CreatePoint(n, m, p, q, B, knotsCsi, knotsEta, csi, eta);
            List<Point3d> dPoint = CreateDerivativePoint(n, m, p, q, B, knotsCsi, knotsEta, csi, eta);
            double N = BasisFunction(knotsCsi, i - 1, p, csi);
            double dN_dCsi = DerivativeBasisFunction(knotsCsi, i - 1, p, csi);
            double M = BasisFunction(knotsEta, j - 1, q, eta);
            double dM_dEta = DerivativeBasisFunction(knotsEta, j - 1, q, eta);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, point);
            DA.SetData(1, dPoint[0]);
            DA.SetData(2, dPoint[1]);
            DA.SetData(3, N);
            DA.SetData(4, dN_dCsi);
            DA.SetData(5, M);
            DA.SetData(6, dM_dEta);

        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<List<Point3d>> CreateControlPointList(List<Point3d> controlPoints, int n, int m)
        {
            List<List<Point3d>> B = new List<List<Point3d>>();

            for (int j = 0; j < m; j++)
            {
                List<Point3d> B_i = new List<Point3d>();
                for (int i = 0; i < n; i++)
                {
                    B_i.Add(controlPoints[i + j * n]);
                }
                B.Add(B_i);
            }

            return B;
        }

        double BasisFunction(List<double> knotsCsi, int i, int p, double csi)
        {
            double sum = 0;

            if (p == 0)
            {
                if (csi >= knotsCsi[i] && csi <= knotsCsi[i + 1])
                {
                    return 1;
                }
                else return 0;
            }
            else
            {
                double d1 = knotsCsi[i + p] - knotsCsi[i];
                double d2 = knotsCsi[i + p + 1] - knotsCsi[i + 1];

                if (d1 != 0 && d2 == 0)
                {
                    double e1 = csi - knotsCsi[i];
                    sum = e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1;
                }
                else if (d1 == 0 && d2 != 0)
                {
                    double e2 = knotsCsi[i + p + 1] - csi;
                    sum = e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2;
                }
                else if (d1 != 0 && d2 != 0)
                {
                    double e1 = csi - knotsCsi[i];
                    double e2 = knotsCsi[i + p + 1] - csi;

                    if (csi == knotsCsi[i + 1])
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1);
                    }
                    else
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1) + (e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2);
                    }
                }
                else return 0;

                return sum;
            }
        }

        double DerivativeBasisFunction(List<double> knotsCsi, int i, int p, double csi)
        {
            double sum = 0;

            if (p == 0)
            {
                if (csi >= knotsCsi[i] && csi <= knotsCsi[i + 1])
                {
                    return 1;
                }
                else return 0;
            }
            else
            {
                double d1 = knotsCsi[i + p] - knotsCsi[i];
                double d2 = knotsCsi[i + p + 1] - knotsCsi[i + 1];

                if (d1 != 0 && d2 == 0)
                {
                    double e1 = p;
                    sum = e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1;
                }
                else if (d1 == 0 && d2 != 0)
                {
                    double e2 = p;
                    sum = -e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2;
                }
                else if (d1 != 0 && d2 != 0)
                {
                    double e1 = p;
                    double e2 = p;

                    if (csi == knotsCsi[i + 1])
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1);
                    }
                    else
                    {
                        sum = (e1 * BasisFunction(knotsCsi, i, p - 1, csi) / d1) - (e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2);
                    }
                }
                else return 0;

                return sum;
            }
        }

        Point3d CreatePoint(int n, int m, int p, int q, List<List<Point3d>> B, List<double> knotsCsi, List<double> knotsEta, double csi, double eta)
        {

            Point3d point = new Point3d(0, 0, 0);
            double N, M;

            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    N = BasisFunction(knotsCsi, i, p, csi);
                    M = BasisFunction(knotsEta, j, q, eta);
                    point += N * M * B[j][i];
                }
            }

            return point;
        }

        List<Point3d> CreateDerivativePoint(int n, int m, int p, int q, List<List<Point3d>> cp, List<double> knots1, List<double> knots2, double csi, double eta)
        {

            Point3d pointCsi = new Point3d(0, 0, 0);
            Point3d pointEta = new Point3d(0, 0, 0);
            double N, dN, M, dM;

            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    N = BasisFunction(knots1, i, p, csi);
                    dN = DerivativeBasisFunction(knots1, i, p, csi);
                    M = BasisFunction(knots2, j, q, eta);
                    dM = DerivativeBasisFunction(knots2, j, q, eta);
                    pointCsi += dN * M * cp[j][i];
                    pointEta += N * dM * cp[j][i];
                }
            }

            return new List<Point3d>() { pointCsi, pointEta };
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                // return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("23f60be0-9daf-4911-a360-b0f15cca3a9e"); }
        }
    }
}