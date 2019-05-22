using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

// In order to load the result of this wizard, you will also need to
// add the output bin/ folder of this project to the list of loaded
// folder in Grasshopper.
// You can use the _GrasshopperDeveloperSettings Rhino command for that.

namespace IGA
{
    public class C_dB : GH_Component
    {

        public C_dB()
          : base("Cox-de Boor", "C_dB",
              "Basis functions",
              "IGA", "07. Test")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter("Knot vector", "knot_vector", "Knot vector", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degree", "p", "Degree", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Index", "i", "Index", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("Csi", "csi", "Csi", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddNumberParameter("Value", "value", "Value from basis function", GH_ParamAccess.item);
            pManager.AddPointParameter("Points", "points", "List of points", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<double> knots = new List<double>();
            int p = 0;
            int i = 1;
            double csi = 0;

            if (!DA.GetDataList(0, knots)) return;
            if (!DA.GetData(1, ref p)) return;
            if (!DA.GetData(2, ref i)) return;
            if (!DA.GetData(3, ref csi)) return;

            if (i >= knots.Count - p)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Index can not be greater than or equal to the number of knots minus the degree");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            double value = BasisFunction(knots, i - 1, p, csi);

            List<Point3d> points = new List<Point3d>();

            for (double k = 0; k <= knots[knots.Count - 1]; k += 0.01)
            {
                points.Add(new Point3d(k, 0, BasisFunction(knots, i - 1, p, k)));
            }

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, value);
            DA.SetDataList(1, points);

        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        double BasisFunction(List<double> knots, int i, int p, double csi)
        {
            double sum = 0;

            if (p == 0)
            {
                if (csi >= knots[i] && csi <= knots[i + 1])
                {
                    return 1;
                }
                else return 0;
            }
            else
            {
                double d1 = knots[i + p] - knots[i];
                double d2 = knots[i + p + 1] - knots[i + 1];

                if (d1 != 0 && d2 == 0)
                {
                    double e1 = csi - knots[i];
                    sum = e1 * BasisFunction(knots, i, p - 1, csi) / d1;
                }
                else if (d1 == 0 && d2 != 0)
                {
                    double e2 = knots[i + p + 1] - csi;
                    sum = e2 * BasisFunction(knots, i + 1, p - 1, csi) / d2;
                }
                else if (d1 != 0 && d2 != 0)
                {
                    double e1 = csi - knots[i];
                    double e2 = knots[i + p + 1] - csi;

                    if (csi == knots[i + 1])
                    {
                        sum = (e1 * BasisFunction(knots, i, p - 1, csi) / d1);
                    }
                    else
                    {
                        sum = (e1 * BasisFunction(knots, i, p - 1, csi) / d1)  + (e2 * BasisFunction(knots, i + 1, p - 1, csi) / d2);
                    }
                }
                else return 0;

                return sum;
            }
        }

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                // You can add image files to your project resources and access them like this:
                //return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("b53fb1d5-62c3-4f17-8964-b9393a7b30d4"); }
        }
    }
}
