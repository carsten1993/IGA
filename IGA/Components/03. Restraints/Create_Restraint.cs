using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA.Components._03._Restraints
{
    public class Create_Restraint : GH_Component
    {

        public Create_Restraint()
          : base("Create Restraint", "create_restraint",
              "Applies retraints to nodes",
              "IGA", "03. Restraints")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddIntegerParameter("Shape functions/control points", "idLists", "List of shape functions/control points to be restrained", GH_ParamAccess.list);
            pManager.AddBooleanParameter("Tx", "tX", "Restraining displacement in the X-direction", GH_ParamAccess.item, false);
            pManager.AddBooleanParameter("Ty", "tY", "Restraining displacement in the Y-direction", GH_ParamAccess.item, false);
            pManager.AddBooleanParameter("Tz", "tZ", "Restraining displacement in the Z-direction", GH_ParamAccess.item, false);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Restraint", "restraint", "Shape functions/control points to be restrained in the X-, Y- and/or Z-direction", GH_ParamAccess.item);
            pManager.AddTextParameter("Info", "info", "Info about the restraint", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<int> idLists = new List<int>();
            bool tX = false;
            bool tY = false;
            bool tZ = false;

            if (!DA.GetDataList(0, idLists)) return;
            if (!DA.GetData(1, ref tX)) return;
            if (!DA.GetData(2, ref tY)) return;
            if (!DA.GetData(3, ref tZ)) return;

            if (idLists.Count <= 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "No nodes are selected");

            }
            if (!tX && !tY && !tZ)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "No restraints are selected");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            List<int> id = Sort(idLists);
            Restraint restraint = new Restraint(id, tX, tY, tZ);
            string info = restraint.GetRestraintInfo();

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, restraint);
            DA.SetData(1, info);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<int> Sort(List<int> idLists)
        {
            List<int> id = new List<int>();
            id.Add(idLists[0]);
            int temp1, temp2;
            int c;

            for (int i = 1; i < idLists.Count; i++)
            {
                temp1 = idLists[i];
                c = 0;

                for (int j = 0; j < id.Count; j++)
                {
                    temp2 = id[j];
                    if (temp1 == temp2)
                    {
                        c++;
                    }
                }

                if (c == 0)
                {
                    id.Add(idLists[i]);
                }
            }

            return id;
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
            get { return new Guid("962734cb-7c94-4c95-990b-5d532764081f"); }
        }
    }
}