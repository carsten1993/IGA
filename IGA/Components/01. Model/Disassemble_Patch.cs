using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Disassemble_Mesh : GH_Component
    {

        public Disassemble_Mesh()
          : base("Disassemble Patch", "disassemble_patch",
              "Disassemble a patch",
              "IGA", "01. model")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Patch", "patch", "Patch", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Info", "info", "Info", GH_ParamAccess.item);
            pManager.AddPointParameter("Control points", "cp", "Control points", GH_ParamAccess.list);
            pManager.AddPointParameter("New control points", "ncp", "Control points of deformed geometry", GH_ParamAccess.list);
            pManager.AddNumberParameter("Knots csi", "knotsCsi", "Knot vector in the csi direction", GH_ParamAccess.list);
            pManager.AddNumberParameter("Knots eta", "knotsEta", "Knot vector in the eta direction", GH_ParamAccess.list);
            pManager.AddNumberParameter("Knots zeta", "knotsZeta", "Knot vector in the zeta direction", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Degrees", "deg", "List with the degrees [p, q, r]", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Number of shape functions", "shapes", "List with the number of shape functions/control points the csi, eta and zeta direction [n, m, l]", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Size", "size", "List with the sizes [nel, nnp, nen]", GH_ParamAccess.list);
            pManager.AddIntegerParameter("IEN", "ien", "IEN-martix [nel, nen] as list", GH_ParamAccess.list);
        }


        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Patch patch = new Patch();

            if (!DA.GetData(0, ref patch)) return;

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            string info = patch.GetMeshInfo();
            List<List<List<Point3d>>> B = patch.GetControlPoints();
            List<Point3d> cp = ThreeListsToList_point3d(B);
            List<List<List<Point3d>>> newB = patch.GetNewControlPoints();
            List<Point3d> ncp = ThreeListsToList_point3d(newB);
            (List<double> KnotsCsi, List<double> KnotsEta, List<double> KnotsZeta) = patch.GetKnotVectors();
            (int p, int q, int r) = patch.GetDegrees();
            List<int> deg = new List<int>() { p, q, r };
            (int n, int m, int l) = patch.GetNML();
            List<int> shapes = new List<int>() { n, m, l };
            (int nel, int nnp, int nen) = patch.GetSize();
            List<int> size = new List<int>() { nel, nnp, nen };
            (List<List<int>> INN, List<List<int>> IEN) = patch.GetINNIEN();
            List<int> ien = TwoListsToList_int(IEN);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, info);
            DA.SetDataList(1, cp);
            DA.SetDataList(2, ncp);
            DA.SetDataList(3, KnotsCsi);
            DA.SetDataList(4, KnotsEta);
            DA.SetDataList(5, KnotsZeta);
            DA.SetDataList(6, deg);
            DA.SetDataList(7, shapes);
            DA.SetDataList(8, size);
            DA.SetDataList(9, ien);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<Point3d> ThreeListsToList_point3d(List<List<List<Point3d>>> iList)
        {
            List<Point3d> oList = new List<Point3d>();

            foreach(List<List<Point3d>> ll in iList)
            {
                foreach (List<Point3d> l in ll)
                {
                    foreach (Point3d p in l)
                    {
                        oList.Add(p);
                    }
                }
            }

            return oList;
        }

        List<int> TwoListsToList_int(List<List<int>> iList)
        {
            List<int> oList = new List<int>();

            foreach (List<int> l in iList)
            {
                foreach (int d in l)
                {
                    oList.Add(d);
                }
            }

            return oList;
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
            get { return new Guid("a38cf1a4-0c35-41da-90e3-05b03f3e14b0"); }
        }
    }
}