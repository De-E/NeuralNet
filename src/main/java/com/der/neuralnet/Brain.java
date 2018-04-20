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
package com.der.neuralnet;

import com.der.neuralnet.inner.Node;
import java.util.ArrayList;

/**
 *
 * @author Raffaele Francesco Mancino
 */
public class Brain
{
    private final static float LEARNING_RATE = 0.05f;
    
    private ArrayList<Node> input = new ArrayList<>();
    private ArrayList<Node> output = new ArrayList<>();
    private ArrayList<Node> hidden = new ArrayList<>();
    
    public Brain(int input, int output, int hidden)
    {
        for (int i=0; i<input; i++)
        {
            this.input.add(new Node());
        }
        for (int i=0; i<output; i++)
        {
            this.output.add(new Node());
        }
        for (int i=0; i<hidden; i++)
        {
            this.hidden.add(new Node());
        }
        this.createEdges();
    }
    
    private void createEdges()
    {
        if(this.hidden.size()!=0)
        {
            for(int i=0; i<this.input.size(); i++)
            {
                for(int j=0; j<this.hidden.size(); j++)
                {
                    this.input.get(i).addNext(this.hidden.get(j));
                }
            }
            for(int i=0; i<this.hidden.size(); i++)
            {
                for(int j=0; j<this.output.size(); j++)
                {
                    this.hidden.get(i).addNext(this.output.get(j));
                }
            }
        }else{
            for(int i=0; i<this.input.size(); i++)
            {
                for(int j=0; j<this.output.size(); j++)
                {
                    this.input.get(i).addNext(this.output.get(j));
                }
            }
        }
    }
}
