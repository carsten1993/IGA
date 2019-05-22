using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Curve_Test : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the Curve_Test class.
        /// </summary>
        public Curve_Test()
          : base("Curve Test", "curve_test",
              "Testing if points are within the curve, and the derivative",
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
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Point", "point", "Point on curve for the given csi", GH_ParamAccess.item);
            pManager.AddPointParameter("Derivative", "dPoint", "The derivative of the curve at the point given by csi", GH_ParamAccess.item);
            pManager.AddNumberParameter("Basis function", "N", "The basis function given the degree, index and csi", GH_ParamAccess.item);
            pManager.AddNumberParameter("Derivative basis function", "dN_dCsi", "The derivative of the basis function w.r.t. csi", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<Point3d> B = new List<Point3d>();
            List<double> knotsCsi = new List<double>();
            int p = 0;
            int i = 1;
            double csi = 0;

            if (!DA.GetDataList(0, B)) return;
            if (!DA.GetDataList(1, knotsCsi)) return;
            if (!DA.GetData(2, ref p)) return;
            if (!DA.GetData(3, ref i)) return;
            if (!DA.GetData(4, ref csi)) return;

            int n = knotsCsi.Count - p - 1;

            if (i > n)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Index csi can not be greater than or equal to the number of knots minus the degree");
                return;
            }
            if (n != B.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Number of control points does not match the number of basis function for the given degree");
                return;
            }
            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Point3d point = CreatePoint(n, p, B, knotsCsi, csi);
            Point3d dPoint = CreateDerivativePoint(n, p, B, knotsCsi, csi);
            double N = BasisFunction(knotsCsi, i - 1, p, csi);
            double dN_dCsi = DerivativeBasisFunction(knotsCsi, i - 1, p, csi);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, point);
            DA.SetData(1, dPoint);
            DA.SetData(2, N);
            DA.SetData(3, dN_dCsi);

        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

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
                    sum = - e2 * BasisFunction(knotsCsi, i + 1, p - 1, csi) / d2;
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

        Point3d CreatePoint(int n, int p, List<Point3d> B, List<double> knots, double csi)
        {

            Point3d point = new Point3d(0, 0, 0);
            double N;

            for (int i = 0; i < n; i++)
            {
                N = BasisFunction(knots, i, p, csi);
                point += N * B[i];
            }

            return point;
        }

        Point3d CreateDerivativePoint(int n, int p, List<Point3d> B, List<double> knots, double csi)
        {

            Point3d point = new Point3d(0, 0, 0);
            double dN;

            for (int i = 0; i < n; i++)
            {
                dN = DerivativeBasisFunction(knots, i, p, csi);
                point += dN * B[i];
            }

            return point;
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
            get { return new Guid("1ec330b8-2f1e-46d1-92c3-19f357f62f61"); }
        }
    }
}