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

/**
 *
 * @author Raffaele Francesco Mancino
 */
public class Node
{
    //output operation
    private ArrayList<Node> next = new ArrayList<>();
    private ArrayList<Float> waight = new ArrayList<>();
    //input operation
    private float value = 0.0f;
    private ArrayList<Float> input = new ArrayList<>();
    
    public void addNext(Node n)
    {
        this.next.add(n);
        this.waight.add(1f);
    }
    
    public float getNextWaight(int i)
    {
        return this.waight.get(i);
    }
    
    public void setNextWaight(int i, float value)
    {
        this.waight.set(i, value);
    }
    
    public void insert(float f)
    {
        this.input.add(f);
    }
    
    public void forward()
    {
        this.value=0f;
        for(Float f : this.input)
        {
            this.value += f;
        }
        for(int i=0; i<this.next.size(); i++)
        {
            Node n = this.next.get(i);
            n.insert(this.value*this.waight.get(i));
        }
    }
}
