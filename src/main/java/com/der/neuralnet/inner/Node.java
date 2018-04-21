/*
 * Copyright (C) 2018 Raffaele Francesco Mancino
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package com.der.neuralnet.inner;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Raffaele Francesco Mancino
 */
public class Node
{
    public int id = 0;
    //output operation
    public ArrayList<Node> nextNodes = new ArrayList<>();
    public ArrayList<Float> weightNextNodes = new ArrayList<>();
    public Float value = 0f;
    //input operation
    public ArrayList<Float> input = new ArrayList<>();
    public ArrayList<Float> backWeight = new ArrayList<>();
    //memory operation
    public Float oldValue = 0f;
    
    public void addNext(Node n)
    {
        Random random = new Random();
        
        this.nextNodes.add(n);
        this.weightNextNodes.add(random.nextFloat());
    }
    
    public void forward()
    {
        this.value = this.out();
        for(int i=0; i<this.nextNodes.size(); i++)
        {
            this.nextNodes.get(i).input.add(this.value*this.weightNextNodes.get(i));
        }
        this.oldValue = this.value;
        this.value=0f;
    }
    
    public float out()
    {
        float ret = 0f;
        for(Float f : this.input)
        {
            ret += f;
        }
        /**
         * Clean input array, else it every run duplicate the input
         */
        this.input = new ArrayList<>();
        return ret;
    }
}
