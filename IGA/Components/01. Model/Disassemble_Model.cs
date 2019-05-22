using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;

namespace IGA.Components._01._Model
{
    public class Disassemble_Model : GH_Component
    {

        public Disassemble_Model()
          : base("Disassemble Model", "disassemble_model",
              "Deconstructs a model; Displays a mesh, material properties, restraints, loads.",
              "IGA", "01. Model")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Model", "model", "An assembled model", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Patch", "patch", "Patch", GH_ParamAccess.item);
            pManager.AddGenericParameter("Material", "material", "Material properties", GH_ParamAccess.item);
            //pManager.AddGenericParameter("Loads", "loads", "External loads", GH_ParamAccess.list);
            pManager.AddGenericParameter("Restraints", "restraints", "Nodal restraints", GH_ParamAccess.list);
            pManager.AddGenericParameter("Full stiffness matrix", "K_full", "Full global stiffness matrix", GH_ParamAccess.item);
            pManager.AddGenericParameter("Full external load vector", "P_full", "Full global force vector", GH_ParamAccess.item);
            pManager.AddGenericParameter("Full self load vector", "P_full", "Full global force vector", GH_ParamAccess.item);
            pManager.AddGenericParameter("Stiffness matrix", "K", "Global stiffness matrix", GH_ParamAccess.item);
            pManager.AddGenericParameter("Force vector", "P", "Global force vector", GH_ParamAccess.item);
            pManager.AddNumberParameter("Nodal displacements", "nodal_displacements", "Nodal displacements", GH_ParamAccess.list);
            pManager.AddPointParameter("New control points", "newB", "New control points as a list", GH_ParamAccess.list);
            //pManager.AddNumberParameter("Nodal strains", "nodal_strains", "Nodal strains", GH_ParamAccess.list);
            //pManager.AddNumberParameter("Nodal stresses", "nodal_stresses", "Nodal stresses", GH_ParamAccess.list);
            //pManager.AddNumberParameter("Mises", "mises", "Mises ", GH_ParamAccess.list);
            pManager.AddTextParameter("Info", "info", "Info", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Model model = new Model();

            if (!DA.GetData(0, ref model)) return;

            Patch patch = model.GetPatch();
            Material material = model.GetMaterial();
            //List<PointLoad> pointLoads = model.GetPointLoads();
            List<Restraint> restraints = model.GetRestraints();
            Vector<double> d = model.GetDisplacementVector();
            Matrix<double> K_full = model.GetFullStiffnessMatrix();
            Vector<double> P0 = model.GetSelfLoadVector();
            Vector<double> P1 = model.GetExternalLoadVector();
            Matrix<double> K = model.GetStiffnessMatrix();
            Vector<double> P = model.GetLoadVector();
            //List<double> strains = model.GetStrains();
            //List<double> stresses = model.GetStresses();
            //List<double> mises = model.GetMises();

            (int nel, int nnp, int nen) = patch.GetSize();
            List<List<List<Point3d>>> newB = patch.GetNewControlPoints();

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////


            List<double> d_list = DisplacementVectorToList(d, nnp);
            List<Point3d> ncp = ThreeListsToList_point3d(newB);

            string meshinfo = patch.GetMeshInfo();
            string materialinfo = material.GetMaterialInfo();
            List<string> resinfo = new List<string>();
            //List<string> loadinfo = new List<string>();
            List<string> info = new List<string>();

            foreach (Restraint restraint in restraints)
            {
                resinfo.Add(restraint.GetRestraintInfo());
            }
            /*
            foreach (PointLoad p in pointLoads)
            {
                loadinfo.Add(p.GetPointLoadInfo());
            }*/

            info.Add(meshinfo);
            info.Add(materialinfo);
            //info.AddRange(loadinfo);
            info.AddRange(resinfo);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, patch);
            DA.SetData(1, material);
            //DA.SetDataList(2, pointLoads);
            DA.SetDataList(2, restraints);
            DA.SetData(3, K_full);
            DA.SetData(4, P1);
            DA.SetData(5, P0);
            DA.SetData(6, K);
            DA.SetData(7, P);
            DA.SetDataList(8, d_list);
            DA.SetDataList(9, ncp);
            DA.SetDataList(10, info);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<double> VectorToList_double(Vector<double> vector)
        {
            List<double> list = new List<double>();
            for (int i = 0; i < vector.Count; i++)
            {
                list.Add(vector[i]);
            }

            return list;
        }

        List<double> DisplacementVectorToList(Vector<double> d, int nnp)
        {
            List<double> list = new List<double>();
            if (d == null)
            {
                for (int i = 0; i < 3 * nnp; i++)
                {
                    list.Add(0);
                }
            }
            else
            {
                list = VectorToList_double(d);
            }

            return list;

        }

        List<Point3d> ThreeListsToList_point3d(List<List<List<Point3d>>> iList)
        {
            List<Point3d> oList = new List<Point3d>();

            foreach (List<List<Point3d>> ll in iList)
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
            get { return new Guid("2cede60d-a25a-41fd-af20-b4e5731e006f"); }
        }
    }
}