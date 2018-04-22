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
    private float LEARNING_RATE = 0.05f;
    
    private ArrayList<Node> inputNode = new ArrayList<>();
    private ArrayList<Node> outputNode = new ArrayList<>();
    private ArrayList<Node> hiddenNode = new ArrayList<>();
    
    public Brain(int input, int output, int hidden)
    {
        int j = 0;
        for (int i=0; i<input; i++)
        {
            this.inputNode.add(new Node());
            this.inputNode.get(this.inputNode.size()-1).id = j;
            j++;
        }
        for (int i=0; i<output; i++)
        {
            this.outputNode.add(new Node());
            this.outputNode.get(this.outputNode.size()-1).id = j;
            j++;
        }
        for (int i=0; i<hidden; i++)
        {
            this.hiddenNode.add(new Node());
            this.hiddenNode.get(this.hiddenNode.size()-1).id = j;
            j++;
        }
        this.createEdges();
    }
    
    /**
     * This function create connection between nodes
     */
    private void createEdges()
    {
        if(this.hiddenNode.size()!=0)
        {
            for(int i=0; i<this.inputNode.size(); i++)
            {
                for(int j=0; j<this.hiddenNode.size(); j++)
                {
                    this.inputNode.get(i).addNext(this.hiddenNode.get(j));
                }
            }
            for(int i=0; i<this.hiddenNode.size(); i++)
            {
                for(int j=0; j<this.outputNode.size(); j++)
                {
                    this.hiddenNode.get(i).addNext(this.outputNode.get(j));
                }
            }
        }else{
            for(int i=0; i<this.inputNode.size(); i++)
            {
                for(int j=0; j<this.outputNode.size(); j++)
                {
                    this.inputNode.get(i).addNext(this.outputNode.get(j));
                }
            }
        }
    }
    
    public void changeLearningRate(float learningRate)
    {
        this.LEARNING_RATE = learningRate;
    }
    
    public void input(ArrayList<Float> inputValues)
    {
        for(int i=0; i<this.inputNode.size(); i++)
        {
            this.inputNode.get(i).input = new ArrayList<>();
            this.inputNode.get(i).input.add(inputValues.get(i));
        }
    }
    
    public void run()
    {
        for(Node n : this.inputNode)
        {
            n.forward();
        }
        for(Node n : this.hiddenNode)
        {
            n.forward();
        }
    }
    
    public ArrayList<Float> output()
    {
        ArrayList<Float> o = new ArrayList<>();
        for(Node n : this.outputNode)
        {
            o.add(n.out());
        }
        return o;
    }
    
    /**
     * Adjust weight on error.
     * @param errors Correct value - obtained value in a vector. The size must
     * be the same of the output node and ordered like this nodes.
     */
    public void wrong(ArrayList<Float> errors)
    {
        for(int i=0; i<this.outputNode.size(); i++)
        {
            Node node = this.outputNode.get(i);
            ArrayList<Node> priors = this.findBackNode(node);
            for(int j=0; j< priors.size(); j++)
            {
                Node prior = priors.get(j);
                int k = this.getIntOfWeight(node, prior);
                Float newWeight = this.LEARNING_RATE*errors.get(i)*prior.oldValue+prior.weightNextNodes.get(k);
                prior.weightNextNodes.set(k, newWeight);
            }
        }
    }
    
    /**
     * Return prior node in flow.
     * @param node to get priors
     * @return list of priors
     */
    private ArrayList<Node> findBackNode(Node node)
    {
        ArrayList<Node> ret = new ArrayList<>();
        for(Node n : this.hiddenNode)
        {
            for(Node nextNode : n.nextNodes)
            {
                if(nextNode.id==node.id)
                {
                    ret.add(n);
                }
            }
        }
        for(Node n : this.inputNode)
        {
            for(Node nextNode : n.nextNodes)
            {
                if(nextNode.id==node.id)
                {
                    ret.add(n);
                }
            }
        }
        return ret;
    }
    
    private void wrongOnNode(Float error, Node node)
    {
        
    }
    
    private Integer getIntOfWeight(Node node, Node prior)
    {
        for(int i=0; i<prior.nextNodes.size(); i++)
        {
            Node priorNext = prior.nextNodes.get(i);
            if(priorNext.id == node.id)
            {
                return i;
            }
        }
        return null;
    }
    
    public String printWeightDebug()
    {
        String ret = "";
        ret += "<------>\n";
        for(Node n : this.inputNode)
        {
            for(Float f : n.weightNextNodes)
                ret += f + " ";
            ret += "\n";
        }
        ret += "<------>\n";
        return ret;
    }
}
