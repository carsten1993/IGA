using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IGA
{
    class Restraint
    {
        private List<int> id;
        private bool resX, resY, resZ;
        private List<List<int>> res;

        public Restraint() { }

        public Restraint(List<int> _id, bool _resX, bool _resY, bool _resZ)
        {
            id = _id;
            resX = _resX;;
            resY = _resY;
            resZ = _resZ;

            int iResX, iResY, iResZ;
            if (resX)
            {
                iResX = 1;
            }
            else
            {
                iResX = 0;
            }
            if (resY)
            {
                iResY = 1;
            }
            else
            {
                iResY = 0;
            }
            if (resZ)
            {
                iResZ = 1;
            }
            else
            {
                iResZ = 0;
            }

            res = new List<List<int>>();
            foreach(int d in id)
            {
                List<int> temp = new List<int>() { d, iResX, iResY, iResZ };
                res.Add(temp);
            }
        }

        public List<int> GetID() { return id; }
        public (bool, bool, bool) GetBooleanRestraints() { return (resX, resY, resZ); }
        public List<List<int>> GetRestraints() { return res; }
        public string GetRestraintInfo()
        {
            if (resX && resY && resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the X, Y and Z-direction";
            }
            else if (resX && resY && !resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the X and Y-direction";
            }
            else if (resX && !resY && resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the X and Z-direction";
            }
            else if (!resX && resY && resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the Y and Z-direction";
            }
            else if (resX && !resY && !resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the X-direction";
            }
            else if (!resX && resY && !resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the Y-direction";
            }
            else if (!resX && !resY && resZ)
            {
                return id.Count.ToString() + " shape functions/control points are restrained in the Z-direction";
            }
            else
            {
                return "There are no retraints";
            }
        }
    }
}
